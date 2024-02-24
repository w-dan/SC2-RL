import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import redis
import redis.asyncio as aioredis
from sc2 import maps  # maps method for loading maps to play in.
from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId

# function that facilitates actually running the agents in games
from sc2.main import run_game

# wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.unit import Unit

import sc2_rl.types.game as game
import sc2_rl.types.game as game_info
import sc2_rl.types.rewards as rewards
from sc2_rl.types.actions import Action


class SupplyException(Exception):
    pass


# Ref: https://github.com/Sentdex/SC2RL
class ArtanisBot(BotAI):
    RACE = Race.Protoss

    def __init__(self, verbose):

        self.verbose = verbose

        self.redis = aioredis.from_url(
            "redis://localhost:6379", encoding="utf-8", decode_responses=True
        )

        self.build_queue: List[rewards.MicroReward] = []
        self.train_queue: List[rewards.MicroReward] = []
        self.scout_list: List[int] = []
        self.rewardMgr = rewards.RewardManager()

    async def on_step(self, iteration: int):
        """`on_step` is a method that is called every step of the game."""
        # self.rewardMgr.apply_scout_reward(
        #     [
        #         (scout, self.units.find_by_tag(scout).is_idle)
        #         for scout in self.scout_list
        #     ]
        # )

        await self.distribute_workers()  # put idle workers back to work

        action = await self.redis.blpop("action_queue", 0)
        action = Action(int(action[1]))

        if action == Action.BUILD_PYLON:
            await self.chat_send("Building Pylon", team_only=True)
            await self.build_pylon(iteration, action)

        elif action == Action.EXPAND:
            await self.chat_send("Expanding", team_only=True)
            try:
                expanded = False
                if self.supply_left < 10:
                    expanded = await self.build_pylon(iteration, action) or expanded

                if not expanded:
                    for nexus in self.townhalls:
                        expanded = self.build_probe(nexus) or expanded
                        expanded = (
                            await self.build_possible_assimilators(nexus) or expanded
                        )

                if not expanded:
                    await self.build_nexus(iteration)

            except Exception as e:
                if self.verbose:
                    print(e)

        elif action == Action.BUILD_STARGATE:
            await self.chat_send("Building Stargate", team_only=True)
            try:
                # iterate thru all nexus and see if these buildings are close
                for nexus in self.townhalls:
                    await self.build_stargate_path(iteration, nexus)

            except Exception as e:
                print(e)

        elif action == Action.TRAIN_VOIDRAY:
            await self.chat_send("Building Voidray", team_only=True)
            try:
                self.build_voidray(iteration)

            except Exception as e:
                if self.verbose:
                    print(e)

        elif action == Action.SCOUT:
            await self.chat_send("Scouting", team_only=True)
            # if self.last_sent doesnt exist yet:
            if not hasattr(self, "last_sent"):
                self.last_sent = 0

            if (iteration - self.last_sent) > game.SCOUT_TIMEOUT:
                try:
                    self.scout(iteration)
                except Exception as e:
                    pass

        elif action == Action.ATTACK:
            await self.chat_send("Attacking", team_only=True)
            try:
                self.attack_path()

            except Exception as e:
                if self.verbose:
                    print(e)

        elif action == Action.FLEE:
            await self.chat_send("Fleeing", team_only=True)
            self.flee()

        state_rwd_action = {
            "state": self._get_state_from_world().tolist(),
            "micro-reward": self._calculate_micro_reward(),
            "info": {
                "game_tick": iteration,
                "n_VOIDRAY": self.units(UnitTypeId.VOIDRAY).amount,
            },
            "game_status": game_info.GameResult.PLAYING.value,
        }
        state_rwd_action = json.dumps(state_rwd_action).encode()

        await self.redis.rpush("state_queue", state_rwd_action)

    async def build_pylon(self, game_tick, action):
        if self.already_pending(UnitTypeId.PYLON) == 0:
            if (
                self.can_afford(UnitTypeId.PYLON)
                and self.townhalls.amount >= game.MIN_NEXUS
            ):
                await self.build(
                    UnitTypeId.PYLON, near=np.random.choice(self.townhalls)
                )
                self.build_queue.append(
                    rewards.MicroReward(game_tick, action, rewards.BUILD_REWARD.PYLON)
                )

                return True

        return False

    def build_probe(self, nexus: Unit):
        # get worker count for this nexus:
        worker_count = len(self.workers.closer_than(game.RANGES.BUILD, nexus))
        if worker_count < (
            game.MAX_WORKERS.NEXUS
            + game.MAX_WORKERS.VESPENE_GEYSER
            + game.MAX_WORKERS.VESPENE_GEYSER
        ):
            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE):
                nexus.train(UnitTypeId.PROBE)
                return True

        return False

    async def build_possible_assimilators(self, nexus: Unit):
        # have we built enough assimilators?
        # find vespene geysers
        expanded = False
        for geyser in self.vespene_geyser.closer_than(game.RANGES.BUILD, nexus):
            try:
                expanded = await self.build_assimilator(geyser) or expanded

            except SupplyException:
                break

        return expanded

    async def build_assimilator(self, geyser: Unit):
        # build assimilator if there isn't one already:
        if self.can_afford(UnitTypeId.ASSIMILATOR):
            raise SupplyException

        if not self.structures(UnitTypeId.ASSIMILATOR).closer_than(2.0, geyser).exists:
            await self.build(UnitTypeId.ASSIMILATOR, geyser)
            return True

        return False

    async def build_nexus(self, game_tick):
        if self.already_pending(UnitTypeId.NEXUS) == 0 and self.can_afford(
            UnitTypeId.NEXUS
        ):
            await self.expand_now()
            self.build_queue.append(
                rewards.MicroReward(
                    game_tick, Action.EXPAND, rewards.BUILD_REWARD.NEXUS
                )
            )

            return True

        return False

    async def build_stargate_path(self, game_tick, nexus: Unit):
        # is there is not a gateway close:
        await self.build_gateway(game_tick, nexus)

        # if the is not a cybernetics core close:
        await self.build_cybernetics(game_tick, nexus)

        # if there is not a stargate close:
        await self.build_stargate(game_tick, nexus)

    async def build_gateway(self, game_tick, nexus: Unit):
        if (
            not self.structures(UnitTypeId.GATEWAY)
            .closer_than(game.RANGES.BUILD, nexus)
            .exists
        ):
            # if we can afford it:
            if (
                self.can_afford(UnitTypeId.GATEWAY)
                and self.already_pending(UnitTypeId.GATEWAY) == 0
            ):
                # build gateway
                await self.build(UnitTypeId.GATEWAY, near=nexus)
                self.build_queue.append(
                    rewards.MicroReward(
                        game_tick, Action.BUILD_STARGATE, rewards.BUILD_REWARD.GATEWAY
                    )
                )

                return True

        return False

    async def build_cybernetics(self, game_tick, nexus: Unit):
        if (
            not self.structures(UnitTypeId.CYBERNETICSCORE)
            .closer_than(game.RANGES.BUILD, nexus)
            .exists
        ):
            # if we can afford it:
            if (
                self.can_afford(UnitTypeId.CYBERNETICSCORE)
                and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
            ):
                # build cybernetics core
                await self.build(UnitTypeId.CYBERNETICSCORE, near=nexus)
                self.build_queue.append(
                    rewards.MicroReward(
                        game_tick,
                        Action.BUILD_STARGATE,
                        rewards.BUILD_REWARD.CYBERNETICSCORE,
                    )
                )

                return True

        return False

    async def build_stargate(self, game_tick, nexus: Unit):
        if (
            not self.structures(UnitTypeId.STARGATE)
            .closer_than(game.RANGES.BUILD, nexus)
            .exists
        ):
            # if we can afford it:
            if (
                self.can_afford(UnitTypeId.STARGATE)
                and self.already_pending(UnitTypeId.STARGATE) == 0
            ):
                # build stargate
                await self.build(UnitTypeId.STARGATE, near=nexus)
                self.build_queue.append(
                    rewards.MicroReward(
                        game_tick, Action.BUILD_STARGATE, rewards.BUILD_REWARD.STARGATE
                    )
                )

                return True

        return False

    def build_voidray(self, game_tick):
        if self.can_afford(UnitTypeId.VOIDRAY):
            for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                if self.can_afford(UnitTypeId.VOIDRAY):
                    sg.train(UnitTypeId.VOIDRAY)
                    self.train_queue.append(
                        rewards.MicroReward(
                            game_tick,
                            Action.TRAIN_VOIDRAY,
                            rewards.TROOPS_REWARD.VOIDRAY_TRAINED,
                        )
                    )

    def scout(self, game_tick: int):
        # are there any idle probes:
        if self.units(UnitTypeId.PROBE).idle.exists:
            # pick one of these randomly:
            probe: Unit = np.random.choice(self.units(UnitTypeId.PROBE).idle)
        else:
            probe: Unit = np.random.choice(self.units(UnitTypeId.PROBE))
        # send probe towards enemy base:
        probe.attack(self.enemy_start_locations[0])
        self.scout_list.append(probe.tag)
        self.rewardMgr.add_reward(
            probe.tag,
            rewards.MicroReward(
                game_tick, Action.SCOUT, rewards.TROOPS_REWARD.PROBE_SCOUTING_STATIC
            ),
        )
        self.rewardMgr.consume_rewards(probe.tag, consume=False)

        self.last_sent = game_tick

    def attack_path(self):
        # take all void rays and attack!
        for voidray in self.units(UnitTypeId.VOIDRAY).idle:
            target = self.select_attack_target(voidray)
            self.perform_attack(voidray, target)

    def select_attack_target(self, voidray: Unit):
        # Enemy units near
        enemy_units_nearby = self.enemy_units.closer_than(game.RANGES.ATTACK, voidray)
        if enemy_units_nearby:
            return np.random.choice(enemy_units_nearby)

        # Enemy structures near
        enemy_structures_nearby = self.enemy_structures.closer_than(
            game.RANGES.ATTACK, voidray
        )
        if enemy_structures_nearby:
            return np.random.choice(enemy_structures_nearby)

        # Enemy units
        if self.enemy_units:
            return np.random.choice(self.enemy_units)

        # Enemy structures
        if self.enemy_structures:
            return np.random.choice(self.enemy_structures)

        # Start location
        if self.enemy_start_locations:
            return self.enemy_start_locations[0]

    def perform_attack(self, voidray: Unit, target: Union[Unit, Point2]):
        if target is not None:
            voidray.attack(target)

    def flee(self):
        if self.units(UnitTypeId.VOIDRAY).amount >= game.MIN_VOIDRAY:
            for vr in self.units(UnitTypeId.VOIDRAY):
                vr.attack(self.start_location)

    def _calculate_micro_reward(self):
        reward = 0.0

        reward += self.rewardMgr.apply_next_tick_rewards()

        try:
            attack_count = 0
            # iterate through our void rays:
            for voidray in self.units(UnitTypeId.VOIDRAY):
                # if voidray is attacking and is in range of enemy unit:
                if voidray.is_attacking and voidray.target_in_range:
                    if self.enemy_units.closer_than(
                        game.RANGES.REWARD, voidray
                    ) or self.enemy_structures.closer_than(game.RANGES.REWARD, voidray):
                        # reward += 0.005 # original was 0.005, decent results, but let's 3x it.
                        reward += 0.015
                        attack_count += 1

        except Exception as e:
            print("reward", e)
            reward = 0

        return reward

    def _get_state_from_world(self):
        if self.verbose:
            print("== WORLD STATE ==")
            print(self.game_info.map_size)
            print("== WORLD STATE ==")

        state_map = np.zeros(
            (self.game_info.map_size[0], self.game_info.map_size[1], 3), dtype=np.uint8
        )

        # draw the minerals
        for mineral in self.mineral_field:
            pos = mineral.position
            c = [175, 255, 255]
            fraction = mineral.mineral_contents / 1800
            if mineral.is_visible:
                if self.verbose:
                    print(mineral.mineral_contents)
                state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    np.uint8(max(0, min(255, int(fraction * i)))) for i in c
                ]
            else:
                state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    20,
                    75,
                    50,
                ]

        # draw the enemy start location:
        for enemy_start_location in self.enemy_start_locations:
            pos = enemy_start_location
            c = [0, 0, 255]
            state_map[math.ceil(pos.y)][math.ceil(pos.x)] = c

        # draw the enemy units:
        for enemy_unit in self.enemy_units:
            pos = enemy_unit.position
            c = [100, 0, 255]
            # get unit health fraction:
            fraction = (
                enemy_unit.health / enemy_unit.health_max
                if enemy_unit.health_max > 0
                else 0.0001
            )
            state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                np.uint8(max(0, min(255, int(fraction * i)))) for i in c
            ]

        # draw the enemy structures:
        for enemy_structure in self.enemy_structures:
            pos = enemy_structure.position
            c = [0, 100, 255]
            # get structure health fraction:
            fraction = (
                enemy_structure.health / enemy_structure.health_max
                if enemy_structure.health_max > 0
                else 0.0001
            )
            state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                np.uint8(max(0, min(255, int(fraction * i)))) for i in c
            ]

        # draw our structures:
        for our_structure in self.structures:
            # if it's a nexus:
            if our_structure.type_id == UnitTypeId.NEXUS:
                pos = our_structure.position
                c = [255, 255, 175]
                # get structure health fraction:
                fraction = (
                    our_structure.health / our_structure.health_max
                    if our_structure.health_max > 0
                    else 0.0001
                )
                state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    np.uint8(max(0, min(255, int(fraction * i)))) for i in c
                ]

            else:
                pos = our_structure.position
                c = [0, 255, 175]
                # get structure health fraction:
                fraction = (
                    our_structure.health / our_structure.health_max
                    if our_structure.health_max > 0
                    else 0.0001
                )
                state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    np.uint8(max(0, min(255, int(fraction * i)))) for i in c
                ]

        # draw the vespene geysers:
        for vespene in self.vespene_geyser:
            # draw these after buildings, since assimilators go over them.
            # tried to denote some way that assimilator was on top, couldnt
            # come up with anything. Tried by positions, but the positions arent identical. ie:
            # vesp position: (50.5, 63.5)
            # bldg positions: [(64.369873046875, 58.982421875), (52.85693359375, 51.593505859375),...]
            pos = vespene.position
            c = [255, 175, 255]
            fraction = vespene.vespene_contents / 2250

            if vespene.is_visible:
                state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    np.uint8(max(0, min(255, int(fraction * i)))) for i in c
                ]
            else:
                state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    50,
                    20,
                    75,
                ]

        # draw our units:
        for our_unit in self.units:
            # if it is a voidray:
            if our_unit.type_id == UnitTypeId.VOIDRAY:
                pos = our_unit.position
                c = [255, 75, 75]
                # get health:
                fraction = (
                    our_unit.health / our_unit.health_max
                    if our_unit.health_max > 0
                    else 0.0001
                )
                state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    np.uint8(max(0, min(255, int(fraction * i)))) for i in c
                ]

            else:
                pos = our_unit.position
                c = [175, 255, 0]
                # get health:
                fraction = (
                    our_unit.health / our_unit.health_max
                    if our_unit.health_max > 0
                    else 0.0001
                )
                state_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    np.uint8(max(0, min(255, int(fraction * i)))) for i in c
                ]

        # return {"map": state_map}
        return state_map

    async def on_building_construction_started(self, unit: Unit):
        if unit.type_id in [
            UnitTypeId.PYLON,
            UnitTypeId.NEXUS,
            UnitTypeId.GATEWAY,
            UnitTypeId.CYBERNETICSCORE,
            UnitTypeId.STARGATE,
        ]:
            for order, reward in enumerate(self.build_queue):
                if reward.action == Action.BUILD_PYLON:
                    if unit.type_id == UnitTypeId.PYLON:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

                elif reward.action == Action.EXPAND:
                    if unit.type_id == UnitTypeId.PYLON:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

                    elif unit.type_id == UnitTypeId.NEXUS:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

                elif reward.action == Action.BUILD_STARGATE:
                    if unit.type_id == UnitTypeId.GATEWAY:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

                    elif unit.type_id == UnitTypeId.CYBERNETICSCORE:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

                    elif unit.type_id == UnitTypeId.STARGATE:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

    async def on_building_construction_complete(self, unit: Unit):
        if unit.type_id in [
            UnitTypeId.PYLON,
            UnitTypeId.NEXUS,
            UnitTypeId.GATEWAY,
            UnitTypeId.CYBERNETICSCORE,
            UnitTypeId.STARGATE,
        ]:
            self.rewardMgr.consume_rewards(unit.tag)

    async def on_unit_created(self, unit: Unit):
        if unit.type_id == UnitTypeId.VOIDRAY:
            self.rewardMgr.add_consume_reward(unit.tag, self.train_queue.pop(0))

    async def on_unit_destroyed(self, unit_tag: int):
        if unit_tag in self.scout_list:
            self.scout_list.remove(unit_tag)
            self.rewardMgr.apply_scout_destroy(unit_tag)


ARTANIS = ArtanisBot(0)
result = run_game(  # run_game is a function that runs the game.
    maps.get("Scorpion_1.01"),  # the map we are playing on
    [
        Bot(
            Race.Protoss, ARTANIS
        ),  # runs our coded bot, protoss race, and we pass our bot object
        Computer(Race.Zerg, Difficulty.Hard),
    ],  # runs a pre-made computer agent, zerg race, with a hard difficulty.
    realtime=False,  # When set to True, the agent is limited in how long each step can take to process.
    disable_fog=False,
)


if str(result) == "Result.Victory":
    game_result = game_info.GameResult.VICTORY
else:
    game_result = game_info.GameResult.DEFEAT

os.makedirs("output", exist_ok=True)
with open("output/results.txt", "a") as f:
    f.write(f"{result}\n")


map_state = np.zeros((224, 224, 3), dtype=np.uint8)
state_rwd_action = {
    "action": None,
    # "state": {"map": map_state},
    "state": map_state.tolist(),
    "micro-reward": 0,
    "info": {
        "game_tick": None,
        "n_VOIDRAY": 0,
    },
    "game_status": game_result.value,
}
state_rwd_action = json.dumps(state_rwd_action).encode()

redis = redis.Redis(host="localhost", port=6379, db=0)
redis.rpush("state_queue", state_rwd_action)


cv2.destroyAllWindows()
cv2.waitKey(1)
time.sleep(3)
sys.exit()
