from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.
from sc2.ids.unit_typeid import UnitTypeId
import random
from log import CustomLogger, LoggerOption

MAP_NAME = "Scorpion_1.01"

class IncrediBot(BotAI): # inhereits from BotAI (part of BurnySC2)
    async def on_step(self, iteration: int): # on_step is a method that is called every step of the game.
        # print(f"{iteration}, n_workers: {self.workers.amount}, n_idle_workers: {self.workers.idle.amount},", \
        #    f"minerals: {self.minerals}, gas: {self.vespene}, cannons: {self.structures(UnitTypeId.PHOTONCANNON).amount},", \
        #    f"pylons: {self.structures(UnitTypeId.PYLON).amount}, nexus: {self.structures(UnitTypeId.NEXUS).amount}", \
        #    f"gateways: {self.structures(UnitTypeId.GATEWAY).amount}, cybernetics cores: {self.structures(UnitTypeId.CYBERNETICSCORE).amount}", \
        #    f"stargates: {self.structures(UnitTypeId.STARGATE).amount}, voidrays: {self.units(UnitTypeId.VOIDRAY).amount}, supply: {self.supply_used}/{self.supply_cap}")
        
        logger = CustomLogger(LoggerOption.FILE, 'IncreditBot')
        logger.info(f"[{iteration}], n_workers: {self.workers.amount}, n_idle_workers: {self.workers.idle.amount}, " \
                    f"minerals: {self.minerals}, gas: {self.vespene}, cannons: {self.structures(UnitTypeId.PHOTONCANNON).amount}, " \
                    f"pylons: {self.structures(UnitTypeId.PYLON).amount}, nexus: {self.structures(UnitTypeId.NEXUS).amount}", \
                    f"gateways: {self.structures(UnitTypeId.GATEWAY).amount}, cybernetics cores: {self.structures(UnitTypeId.CYBERNETICSCORE).amount}", \
                    f"stargates: {self.structures(UnitTypeId.STARGATE).amount}, voidrays: {self.units(UnitTypeId.VOIDRAY).amount}, supply: {self.supply_used}/{self.supply_cap}"
                )

        print(f"[+] Iteration {iteration}")

        await self.distribute_workers()

        # begin logic:
        if self.townhalls:
            nexus = self.townhalls.random
            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE):  
                nexus.train(UnitTypeId.PROBE)

            # if we dont have *any* pylons, we'll build one close to the nexus.
            elif not self.structures(UnitTypeId.PYLON) and self.already_pending(UnitTypeId.PYLON) == 0:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexus)

            elif self.structures(UnitTypeId.PYLON).amount < 5:
                if self.can_afford(UnitTypeId.PYLON):
                    # build from the closest pylon towards the enemy
                    target_pylon = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])
                    # build as far away from target_pylon as possible:
                    pos = target_pylon.position.towards(self.enemy_start_locations[0], random.randrange(8, 15))
                    await self.build(UnitTypeId.PYLON, near=pos)
                    
            elif not self.structures(UnitTypeId.FORGE):
                if self.can_afford(UnitTypeId.FORGE):
                    await self.build(UnitTypeId.FORGE, near=self.structures(UnitTypeId.PYLON).closest_to(nexus))

            elif self.structures(UnitTypeId.FORGE).ready and self.structures(UnitTypeId.PHOTONCANNON).amount < 3:
                if self.can_afford(UnitTypeId.PHOTONCANNON):
                    await self.build(UnitTypeId.PHOTONCANNON, near=nexus)

        else:
            if self.can_afford(UnitTypeId.NEXUS):  # can we afford one?
                await self.expand_now()  # build one!



run_game(  # run_game is a function that runs the game.
    maps.get(MAP_NAME), # the map we are playing on
    [Bot(Race.Protoss, IncrediBot()), # runs our coded bot, protoss race, and we pass our bot object 
     Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
    realtime=False, # When set to True, the agent is limited in how long each step can take to process.
)