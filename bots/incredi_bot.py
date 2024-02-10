from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.ids.unit_typeid import UnitTypeId
import random
from log import CustomLogger, LoggerOption

# Bot from third video of sentdex
# https://www.youtube.com/watch?v=FyhxoHKFIiY&list=PLQVvvaa0QuDcBby2qVDsDv41GghEQfr5E&index=3
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
                    f"pylons: {self.structures(UnitTypeId.PYLON).amount}, nexus: {self.structures(UnitTypeId.NEXUS).amount}, " \
                    f"gateways: {self.structures(UnitTypeId.GATEWAY).amount}, cybernetics cores: {self.structures(UnitTypeId.CYBERNETICSCORE).amount}, " \
                    f"stargates: {self.structures(UnitTypeId.STARGATE).amount}, voidrays: {self.units(UnitTypeId.VOIDRAY).amount}, supply: {self.supply_used}/{self.supply_cap}"
                )

        print(f"[+] Iteration {iteration}")

        await self.distribute_workers()

        # begin logic:
        if self.townhalls:
            nexus = self.townhalls.random

            if self.structures(UnitTypeId.VOIDRAY).amount < 10 and self.can_afford(UnitTypeId.VOIDRAY):
                for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                    sg.train(UnitTypeId.VOIDRAY)

            supply_remaining = self.supply_cap - self.supply_used
            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE) and supply_remaining > 8:  
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

            elif self.structures(UnitTypeId.ASSIMILATOR).amount <= 1:
                for nexus in self.structures(UnitTypeId.NEXUS):
                    vespenes = self.vespene_geyser.closer_than(15, nexus)
                    for vespene in vespenes:
                        if self.can_afford(UnitTypeId.ASSIMILATOR) and not self.already_pending(UnitTypeId.ASSIMILATOR):
                            await self.build(UnitTypeId.ASSIMILATOR, vespene)

            elif not self.structures(UnitTypeId.FORGE):
                if self.can_afford(UnitTypeId.FORGE):
                    await self.build(UnitTypeId.FORGE, near=self.structures(UnitTypeId.PYLON).closest_to(nexus))

            elif self.structures(UnitTypeId.FORGE).ready and self.structures(UnitTypeId.PHOTONCANNON).amount < 3:
                if self.can_afford(UnitTypeId.PHOTONCANNON):
                    await self.build(UnitTypeId.PHOTONCANNON, near=nexus)


            buildings = [UnitTypeId.GATEWAY, UnitTypeId.CYBERNETICSCORE, UnitTypeId.STARGATE]
            for building in buildings:
                if not self.structures(building):
                    if self.can_afford(building):
                        await self.build(building, near=self.structures(UnitTypeId.PYLON).closest_to(nexus))
                        break

        else:
            if self.can_afford(UnitTypeId.NEXUS):  # can we afford one?
                await self.expand_now()  # build one!

        if self.units(UnitTypeId.VOIDRAY).amount >= 5:
            if self.enemy_units:
                for vr in self.units(UnitTypeId.VOIDRAY).idle:
                    vr.attack(random.choice(self.enemy_units))

            elif self.enemy_structures:
                for vr in self.units(UnitTypeId.VOIDRAY).idle:
                    vr.attack(random.choice(self.enemy_structures))

            else:
                for vr in self.units(UnitTypeId.VOIDRAY).idle:
                    vr.attack(self.enemy_start_locations[0])
