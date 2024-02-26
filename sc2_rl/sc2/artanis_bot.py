import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import obswebsocket
import redis
import redis.asyncio as aioredis
from obswebsocket import obsws, requests
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
from sc2_rl.types.colors import ColorPalette


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
            if self.supply_left < 15:
                await self.chat_send("Building Pylon", team_only=True)
                if await self.build_pylon():
                    self.build_queue.append(
                        rewards.MicroReward(
                            iteration, action, rewards.BUILD_REWARD.PYLON
                        )
                    )

                else:
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration, action, -rewards.BUILD_REWARD.PYLON
                        )
                    )

            else:
                self.rewardMgr.apply_unsuccessfull_action(
                    rewards.MicroReward(
                        iteration, action, -2 * rewards.BUILD_REWARD.PYLON
                    )
                )

        elif action == Action.TRAIN_PROBE:
            await self.chat_send("Training Probe", team_only=True)
            try:
                for nexus in self.townhalls:
                    if not self.build_probe(nexus):
                        self.rewardMgr.apply_unsuccessfull_action(
                            rewards.MicroReward(
                                iteration, action, rewards.TROOPS_REWARD.MAX_PROBES
                            )
                        )

            except Exception as e:
                if self.verbose:
                    print(e)

        elif action == Action.BUILD_ASSIMILATOR:
            await self.chat_send("Building Assimilator", team_only=True)
            built = False
            for nexus in self.townhalls:
                for geyser in self.vespene_geyser.closer_than(game.RANGES.BUILD, nexus):
                    built = await self.build_assimilator(geyser) or built

            if not built:
                self.rewardMgr.apply_unsuccessfull_action(
                    rewards.MicroReward(
                        iteration,
                        action,
                        rewards.BUILD_REWARD.ASSIMILATOR_NOT_BUILT,
                    )
                )

        elif action == Action.EXPAND:
            await self.chat_send("Expanding", team_only=True)
            try:
                if await self.build_nexus(iteration):
                    self.build_queue.append(
                        rewards.MicroReward(
                            iteration, Action.EXPAND, rewards.BUILD_REWARD.NEXUS
                        )
                    )

                else:
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration, Action.EXPAND, -rewards.BUILD_REWARD.NEXUS
                        )
                    )

            except Exception as e:
                if self.verbose:
                    print(e)

        elif action == Action.BUILD_GATEWAY:
            await self.chat_send("Building Gateway", team_only=True)
            try:
                all_closer = []
                built = False
                for nexus in self.townhalls:
                    if (
                        not self.structures(UnitTypeId.GATEWAY)
                        .closer_than(game.RANGES.BUILD, nexus)
                        .exists
                    ):
                        all_closer.append(False)
                        built = await self.build_gateway(nexus)

                        if built:
                            break

                    else:
                        all_closer.append(True)

                if built:
                    self.build_queue.append(
                        rewards.MicroReward(
                            iteration,
                            action,
                            rewards.BUILD_REWARD.GATEWAY,
                        )
                    )

                elif all(all_closer):
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration,
                            action,
                            -2 * rewards.BUILD_REWARD.GATEWAY,
                        )
                    )

                elif not built:
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration,
                            action,
                            -rewards.BUILD_REWARD.GATEWAY,
                        )
                    )

            except Exception as e:
                if self.verbose:
                    print(e)

        elif action == Action.BUILD_CYBERNETICSCORE:
            await self.chat_send("Building Cyberneticscore", team_only=True)
            try:
                all_closer = []
                built = False
                for nexus in self.townhalls:
                    if (
                        not self.structures(UnitTypeId.CYBERNETICSCORE)
                        .closer_than(game.RANGES.BUILD, nexus)
                        .exists
                    ):
                        all_closer.append(False)
                        built = await self.build_cybernetics(nexus)

                        if built:
                            break

                    else:
                        all_closer.append(True)

                if built:
                    self.build_queue.append(
                        rewards.MicroReward(
                            iteration,
                            action,
                            rewards.BUILD_REWARD.CYBERNETICSCORE,
                        )
                    )

                elif all(all_closer):
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration,
                            action,
                            -2 * rewards.BUILD_REWARD.CYBERNETICSCORE,
                        )
                    )

                elif not built:
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration,
                            action,
                            -rewards.BUILD_REWARD.CYBERNETICSCORE,
                        )
                    )

            except Exception as e:
                if self.verbose:
                    print(e)

        elif action == Action.BUILD_STARGATE:
            await self.chat_send("Building Stargate", team_only=True)
            try:
                all_closer = []
                built = False
                for nexus in self.townhalls:
                    if (
                        not self.structures(UnitTypeId.STARGATE)
                        .closer_than(game.RANGES.BUILD, nexus)
                        .exists
                    ):
                        all_closer.append(False)
                        built = await self.build_stargate(nexus)

                        if built:
                            break

                    else:
                        all_closer.append(True)

                if built:
                    self.build_queue.append(
                        rewards.MicroReward(
                            iteration,
                            action,
                            rewards.BUILD_REWARD.STARGATE,
                        )
                    )

                elif all(all_closer):
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration,
                            action,
                            -2 * rewards.BUILD_REWARD.STARGATE,
                        )
                    )

                elif not built:
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration,
                            action,
                            -rewards.BUILD_REWARD.STARGATE,
                        )
                    )

            except Exception as e:
                if self.verbose:
                    print(e)

        elif action == Action.TRAIN_VOIDRAY:
            await self.chat_send("Training Voidray", team_only=True)
            try:
                idle_stargates = (
                    self.structures(UnitTypeId.STARGATE).ready.idle.amount > 0
                )
                trained = False
                for stargate in self.structures(UnitTypeId.STARGATE).ready.idle:
                    if self.train_voidray(stargate):
                        self.train_queue.append(
                            rewards.MicroReward(
                                iteration,
                                action,
                                rewards.TROOPS_REWARD.VOIDRAY_TRAINED,
                            )
                        )

                if idle_stargates and not trained:
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration,
                            action,
                            -rewards.TROOPS_REWARD.VOIDRAY_TRAINED,
                        )
                    )

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

            else:
                self.rewardMgr.apply_unsuccessfull_action(
                    rewards.MicroReward(
                        iteration,
                        Action.SCOUT,
                        -rewards.TROOPS_REWARD.PROBE_SCOUTING_STATIC,
                    )
                )

        elif action == Action.ATTACK:
            await self.chat_send("Attacking", team_only=True)
            try:
                if not self.attack_path():
                    self.rewardMgr.apply_unsuccessfull_action(
                        rewards.MicroReward(
                            iteration,
                            Action.ATTACK,
                            rewards.TROOPS_REWARD.NO_VOIDRAY_ATTACK,
                        )
                    )

            except Exception as e:
                if self.verbose:
                    print(e)

        elif action == Action.FLEE:
            await self.chat_send("Fleeing", team_only=True)
            if not self.flee():
                self.rewardMgr.apply_unsuccessfull_action(
                    rewards.MicroReward(
                        iteration,
                        Action.ATTACK,
                        rewards.TROOPS_REWARD.NO_VOIDRAY_ATTACK,
                    )
                )

        state_rwd_action = {
            "state": {
                "structures_state": self._get_structures_map().tolist(),
                "units_state": self._get_units_map().tolist(),
                "minerals": self.minerals,
                "vespene": self.vespene,
                "supply_used": self.supply_used,
                "supply_cap": self.supply_cap,
            },
            "micro-reward": self._calculate_micro_reward(),
            "info": {
                "game_tick": iteration,
                "n_nexus": self.townhalls.amount,
                "n_voidray": self.units(UnitTypeId.VOIDRAY).amount,
                "n_structures": self.structures.amount,
            },
            "game_status": game_info.GameResult.PLAYING.value,
        }
        state_rwd_action = json.dumps(state_rwd_action).encode()

        await self.redis.rpush("state_queue", state_rwd_action)

    async def build_pylon(self):
        if self.already_pending(UnitTypeId.PYLON) == 0:
            if (
                self.can_afford(UnitTypeId.PYLON)
                and self.townhalls.amount >= game.MIN_NEXUS
            ):
                await self.build(
                    UnitTypeId.PYLON, near=np.random.choice(self.townhalls)
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

    async def build_assimilator(self, geyser: Unit):
        # build assimilator if there isn't one already:
        if self.can_afford(UnitTypeId.ASSIMILATOR):
            if (
                not self.structures(UnitTypeId.ASSIMILATOR)
                .closer_than(2.0, geyser)
                .exists
            ):
                await self.build(UnitTypeId.ASSIMILATOR, geyser)
                return True

        return False

    async def build_nexus(self, reconstruct_main=False):
        if self.already_pending(UnitTypeId.NEXUS) == 0 and self.can_afford(
            UnitTypeId.NEXUS
        ):
            location = self.start_location if reconstruct_main else None
            await self.expand_now(location=location)
            return True

        return False

    async def build_gateway(self, nexus: Unit):
        # if we can afford it:
        if (
            self.can_afford(UnitTypeId.GATEWAY)
            and self.already_pending(UnitTypeId.GATEWAY) == 0
        ):
            # build gateway
            await self.build(UnitTypeId.GATEWAY, near=nexus)
            return True

        return False

    async def build_cybernetics(self, game_tick, nexus: Unit):
        # if we can afford it:
        if (
            self.can_afford(UnitTypeId.CYBERNETICSCORE)
            and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
        ):
            # build cybernetics core
            await self.build(UnitTypeId.CYBERNETICSCORE, near=nexus)
            return True

        return False

    async def build_stargate(self, game_tick, nexus: Unit):
        # if we can afford it:
        if (
            self.can_afford(UnitTypeId.STARGATE)
            and self.already_pending(UnitTypeId.STARGATE) == 0
        ):
            # build stargate
            await self.build(UnitTypeId.STARGATE, near=nexus)
            return True

        return False

    def train_voidray(self, stargate):
        if self.can_afford(UnitTypeId.VOIDRAY):
            stargate.train(UnitTypeId.VOIDRAY)
            return True

        return False

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
        self.last_sent = game_tick
        self.rewardMgr.add_reward(
            probe.tag,
            rewards.MicroReward(
                game_tick, Action.SCOUT, rewards.TROOPS_REWARD.PROBE_SCOUTING_STATIC
            ),
        )
        self.rewardMgr.consume_rewards(probe.tag, consume=False)

    def attack_path(self):
        # take all void rays and attack!
        if self.units(UnitTypeId.VOIDRAY).exists:
            for voidray in self.units(UnitTypeId.VOIDRAY).idle:
                target = self.select_attack_target(voidray)
                self.perform_attack(voidray, target)
            return True

        return False

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
            return True

        return False

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

    def _get_structures_map(self):
        structures_map = np.zeros(
            (self.game_info.map_size[0], self.game_info.map_size[1], 3), dtype=np.uint8
        )

        # draw available tiles
        # for y in range(self.game_info.placement_grid.data_numpy.shape[1]):
        #     for x in range(self.game_info.placement_grid.data_numpy.shape[0]):
        # for x in range(self.game_info.placement_grid.height):
        #     for y in range(self.game_info.placement_grid.width):
        #         if self.game_info.placement_grid.is_set((x, y)):
        #             structures_map[y][x] = ColorPalette.GROUND.FREE
        #         else:
        #             structures_map[y][x] = ColorPalette.GROUND.OCCUPIED

        # draw the minerals
        for mineral in self.mineral_field:
            pos = mineral.position
            c = ColorPalette.RESOURCE.MINERAL
            fraction = mineral.mineral_contents / 1800
            if mineral.is_visible:
                if self.verbose:
                    print(mineral.mineral_contents)
                structures_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    np.uint8(max(0, min(255, int(fraction * i)))) for i in c
                ]
            else:
                structures_map[math.ceil(pos.y)][
                    math.ceil(pos.x)
                ] = ColorPalette.RESOURCE.MINERAL_NOT_VISIBLE

        # draw the enemy start location:
        for enemy_start_location in self.enemy_start_locations:
            pos = enemy_start_location
            c = ColorPalette.BUILD.GENERIC
            structures_map[math.ceil(pos.y)][math.ceil(pos.x)] = c

        # draw the enemy structures:
        for enemy_structure in self.enemy_structures:
            pos = enemy_structure.position
            c = ColorPalette.BUILD.ENEMY_STRUCTURE
            # get structure health fraction:
            fraction = (
                enemy_structure.health / enemy_structure.health_max
                if enemy_structure.health_max > 0
                else 0.0001
            )
            structures_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                np.uint8(max(0, min(255, int(fraction * i)))) for i in c
            ]

        # draw our structures:
        for our_structure in self.structures:
            pos = our_structure.position

            # get structure health fraction:
            fraction = (
                our_structure.health / our_structure.health_max
                if our_structure.health_max > 0
                else 0.0001
            )

            # if it's a nexus:
            if our_structure.type_id == UnitTypeId.NEXUS:
                c = ColorPalette.BUILD.NEXUS

            elif our_structure.type_id == UnitTypeId.PYLON:
                c = ColorPalette.BUILD.PYLON

            elif our_structure.type_id == UnitTypeId.ASSIMILATOR:
                c = ColorPalette.BUILD.ASSIMILATOR

            elif our_structure.type_id == UnitTypeId.GATEWAY:
                c = ColorPalette.BUILD.GATEWAY

            elif our_structure.type_id == UnitTypeId.CYBERNETICSCORE:
                c = ColorPalette.BUILD.CYBERNETICSCORE

            elif our_structure.type_id == UnitTypeId.STARGATE:
                c = ColorPalette.BUILD.STARGATE

            else:
                c = ColorPalette.BUILD.GENERIC

            structures_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
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
            c = ColorPalette.RESOURCE.VESPENE
            fraction = vespene.vespene_contents / 2250

            if vespene.is_visible:
                structures_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                    np.uint8(max(0, min(255, int(fraction * i)))) for i in c
                ]
            else:
                structures_map[math.ceil(pos.y)][
                    math.ceil(pos.x)
                ] = ColorPalette.RESOURCE.VESPENE_NOT_VISIBLE

        return structures_map

    def _get_units_map(self):
        units_map = np.zeros(
            (self.game_info.map_size[0], self.game_info.map_size[1], 3), dtype=np.uint8
        )

        # draw available tiles
        # self.game_info.pathing_grid.print()
        # for y in range(self.game_info.pathing_grid.height):
        #     for x in range(self.game_info.placement_grid.width):
        #         if self.game_info.placement_grid.is_set((x, y)):
        #             units_map[y][x] = ColorPalette.GROUND.FREE
        #         else:
        #             units_map[y][x] = ColorPalette.GROUND.OCCUPIED

        # draw the enemy units:
        for enemy_unit in self.enemy_units:
            pos = enemy_unit.position
            c = ColorPalette.UNIT.ENEMY
            # get unit health fraction:
            fraction = (
                enemy_unit.health / enemy_unit.health_max
                if enemy_unit.health_max > 0
                else 0.0001
            )
            units_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                np.uint8(max(0, min(255, int(fraction * i)))) for i in c
            ]

        # draw our units:
        for our_unit in self.units:
            pos = our_unit.position

            if our_unit.type_id == UnitTypeId.VOIDRAY:
                c = ColorPalette.UNIT.VOIDRAY

            elif our_unit.type_id == UnitTypeId.PROBE:
                c = ColorPalette.UNIT.PROBE

            else:
                c = ColorPalette.UNIT.GENERIC
            # get health:
            fraction = (
                our_unit.health / our_unit.health_max
                if our_unit.health_max > 0
                else 0.0001
            )
            units_map[math.ceil(pos.y)][math.ceil(pos.x)] = [
                np.uint8(max(0, min(255, int(fraction * i)))) for i in c
            ]
        return units_map

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

                elif reward.action == Action.BUILD_ASSIMILATOR:
                    if unit.type_id == UnitTypeId.ASSIMILATOR:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

                elif reward.action == Action.EXPAND:
                    if unit.type_id == UnitTypeId.NEXUS:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

                elif reward.action == Action.BUILD_GATEWAY:
                    if unit.type_id == UnitTypeId.GATEWAY:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

                elif reward.action == Action.BUILD_CYBERNETICSCORE:
                    if unit.type_id == UnitTypeId.CYBERNETICSCORE:
                        self.rewardMgr.add_reward(unit.tag, reward)
                        self.build_queue.pop(order)
                        break

                elif reward.action == Action.BUILD_STARGATE:
                    if unit.type_id == UnitTypeId.STARGATE:
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


# # Configura los detalles de conexión a OBS
# host = "192.168.1.54"  # te boi ajiackear >.<
# port = 4455
# password = "8Ytf9TfVlFULRblI"


# client = obswebsocket.obsws(host, port, password)
# client.connect()
# client.call(requests.StartRecord())


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

# client.call(requests.StopRecord())
# client.disconnect()

if str(result) == "Result.Victory":
    game_result = game_info.GameResult.VICTORY

else:
    game_result = game_info.GameResult.DEFEAT

    # folder = "C:\\Users\\User\\Desktop\\UPM\\Master\\RLGAN\\SC2-RL\\output\\replays"
    # for filename in os.listdir(folder):
    #     file_path = os.path.join(folder, filename)
    #     os.remove(file_path)

os.makedirs("output", exist_ok=True)
with open("output/results.txt", "a") as f:
    f.write(f"{result}\n")


structures_map_state = np.zeros((224, 224, 3), dtype=np.uint8)
units_map_state = np.zeros((224, 224, 3), dtype=np.uint8)

state_rwd_action = {
    "state": {
        "structures_state": structures_map_state.tolist(),
        "units_state": units_map_state.tolist(),
        "minerals": 0,
        "vespene": 0,
        "supply_used": 0,
        "supply_cap": 0,
    },
    "micro-reward": 0,
    "info": {
        "game_tick": None,
        "n_nexus": 0,
        "n_voidray": 0,
        "n_structures": 0,
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
