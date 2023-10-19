"""
Created on Fri Oct 7 2022
@author: ynomura
"""

import random
import time


class Character:
    def __init__(self, name, max_hp, max_mp, power, defense):
        self.name = name
        self.max_hp = max_hp
        self.max_mp = max_mp
        self.power = power
        self.defense = defense
        self.hp = self.max_hp
        self.mp = self.max_mp

    # players and monsters are list of object
    def battle(self, players, monsters):
        all_characters = players + monsters
        random.shuffle(all_characters)
        battle_num = 0
        num_players = len(players)
        num_monsters = len(monsters)
        min_player_hp_list = sorted(players[:], key=lambda player: player.hp)
        min_monster_hp_list = sorted(monsters[:], key=lambda monster: monster.hp)

        # The battle ends when one of them is defeated / fainted.
        # attack character of minimum hp
        battle_count = 1
        print('魔物の群れが現れた！')
        while True:
            print(f'- ターン: {battle_count} -')
            if isinstance(all_characters[battle_num], Player):
                # Determine whether the Brave used supporting magic
                is_used_support_magic = False
                if isinstance(all_characters[battle_num], Brave):
                    is_used_support_magic = all_characters[battle_num].support(min_player_hp_list[0])

                # when defeated
                if not is_used_support_magic and all_characters[battle_num].attack(min_monster_hp_list[0]):
                    all_characters.remove(min_monster_hp_list[0])
                    min_monster_hp_list.remove(min_monster_hp_list[0])
                    num_monsters -= 1
                    # when all monsters are defeated
                    if num_monsters <= 0:
                        print('魔物の群れを倒した！')
                        break
            else:
                if all_characters[battle_num].attack(min_player_hp_list[0]):
                    all_characters.remove(min_player_hp_list[0])
                    min_player_hp_list.remove(min_player_hp_list[0])
                    num_players -= 1
                    if num_players <= 0:
                        print('魔物たちに敗北してしまった．')
                        break
            battle_num += 1
            battle_count += 1
            if battle_num >= len(all_characters):
                battle_num = 0
            # for proceeding slowly
            time.sleep(0.5)

    # return true: target is defeated / fainted
    def attack(self, target):
        is_used_method = False
        # magic or special method
        if self.mp > 0 and random.random() < 1 / 3:
            amount_of_damage = 0
            if isinstance(self, Wizard):
                amount_of_damage = self.magic()
                print(f'{self.name}の攻撃！ {self.name} は {self.magic_name} を使った．')
                print(f'{target.name} は {amount_of_damage} のダメージを受けた．')
                target.hp -= amount_of_damage
                is_used_method = True

            elif isinstance(self, Dragon):
                amount_of_damage = self.special()
                print(f'{self.name}の攻撃！ {self.name} は {self.special_name} を使った．')
                print(f'{target.name} は {amount_of_damage} のダメージを受けた．')
                target.hp -= amount_of_damage
                is_used_method = True

        if not is_used_method:
            amount_of_damage = self.power - target.defense // 2
            target.hp -= amount_of_damage
            print(f'{self.name} の攻撃！ {target.name} は {amount_of_damage} のダメージを受けた.')

        if target.hp <= 0:
            if isinstance(target, Player):
                print(f'{target.name} は気絶した！')
            else:
                print(f'{target.name} を倒した！')
            return True
        else:
            return False

    # Supporting magic
    # return True: supporting method is used
    def support(self, supporting_target):
        if self.mp > 0 and random.random() < 1 / 3:
            amount_of_heel = self.magic()
            supporting_target.hp += amount_of_heel
            print(f'{self.name} の攻撃！ {self.name} は {self.magic_name} を使った．')
            print(f' {supporting_target.name} のHPが {amount_of_heel} 回復した．')
            return True
        else:
            return False

    # battle start
    def start_battle(self):
        brave = Brave()
        knight = Knight()
        wizard = Wizard()
        players = [brave, knight, wizard]

        # summon monsters randomly
        summoned_monsters = []
        summoned_monsters_num = random.randint(1, 4)
        for _ in range(summoned_monsters_num):
            monster_index = random.randint(0, 2)
            if monster_index == 0:
                new_monster = Slime()
            elif monster_index == 1:
                new_monster = Mummy()
            elif monster_index == 2:
                new_monster = Dragon()
            summoned_monsters.append(new_monster)

        # Number of times each monster appears
        count_slime = 0
        count_mummy = 0
        count_dragon = 0
        for summoned_monster in summoned_monsters:
            if summoned_monster.name == 'スライム':
                count_slime += 1
            elif summoned_monster.name == 'ミイラ':
                count_mummy += 1
            elif summoned_monster.name == 'ドラゴン':
                count_dragon += 1

        # Exclude monsters that have been summoned only once
        count_slime = count_slime if count_slime > 1 else 0
        count_mummy = count_mummy if count_mummy > 1 else 0
        count_dragon = count_dragon if count_dragon > 1 else 0

        # Naming monsters that summon multiple times
        naming_list = ['A', 'B', 'C', 'D']
        slime_index = 0
        mummy_index = 0
        dragon_index = 0

        for summoned_monster in summoned_monsters:
            if summoned_monster.name == 'スライム':
                if count_slime > 0:
                    summoned_monster.name = f'{summoned_monster.name}{naming_list[slime_index]}'
                    count_slime -= 1
                    slime_index += 1
            elif summoned_monster.name == 'ミイラ':
                if count_mummy > 0:
                    summoned_monster.name = f'{summoned_monster.name}{naming_list[mummy_index]}'
                    count_mummy -= 1
                    mummy_index += 1
            elif summoned_monster.name == 'ドラゴン':
                if count_dragon > 0:
                    summoned_monster.name = f'{summoned_monster.name}{naming_list[dragon_index]}'
                    count_dragon -= 1
                    dragon_index += 1

        # Start a battle
        self.battle(players, summoned_monsters)

    def start_battle_Demon_King(self):
        brave = Brave()
        knight = Knight()
        wizard = Wizard()
        num_players = 3
        players = [brave, knight, wizard]
        devil = DemonKing()
        all_characters = players + [devil]
        random.shuffle(all_characters)
        character_index = 0
        battle_count = 1
        min_player_hp_list = sorted(players, key=lambda player: player.hp)

        print('魔王が現れた！')
        while True:
            print(f'- ターン: {battle_count} -')
            if isinstance(all_characters[character_index], Player):
                # Determine whether the Brave used supporting magic
                is_used_support_magic = False
                if isinstance(all_characters[character_index], Brave):
                    is_used_support_magic = all_characters[character_index].support(min_player_hp_list[0])

                # when defeated
                if not is_used_support_magic and all_characters[character_index].attack(devil):
                    # when all monsters are defeated
                    break

            # Demon King attack
            else:
                is_used_special = False
                for _ in range(2):
                    is_used_special = devil.special(players)
                    if is_used_special:
                        # check HP of all players
                        for player in players:
                            if player.hp <= 0:
                                all_characters.remove(min_player_hp_list[0])
                                players.remove(min_player_hp_list[0])
                                min_player_hp_list.remove(min_player_hp_list[0])
                                num_players -= 1
                        if num_players <= 0:
                            break

                    # normal attack
                    else:
                        # If player faint
                        if devil.attack(min_player_hp_list[0]):
                            all_characters.remove(min_player_hp_list[0])
                            players.remove(min_player_hp_list[0])
                            min_player_hp_list.remove(min_player_hp_list[0])
                            num_players -= 1
                        if num_players <= 0:
                            break

                    # for proceeding slowly
                    time.sleep(0.5)
            if num_players <= 0:
                print('魔王に敗北してしまった．')
                break
            character_index += 1
            battle_count += 1
            if character_index >= len(all_characters):
                character_index = 0


class Player(Character):
    def __init__(self, name, max_hp, max_mp, power, defense):
        super().__init__(name, max_hp, max_mp, power, defense)
        self.magic_name = None

    def magic(self):
        self.mp -= 1


class Monster(Character):
    def __init__(self, name, max_hp, max_mp, power, defense):
        super().__init__(name, max_hp, max_mp, power, defense)
        self.special_name = None

    def special(self):
        self.mp -= 1


class Brave(Player):
    def __init__(self):
        super().__init__("勇者", 100, 3, 45, 30)
        self.magic_name = None

    def magic(self):
        self.magic_name = '回復魔法'
        self.mp -= 1
        amount_of_heel = 15
        return amount_of_heel


class Knight(Player):
    def __init__(self):
        super().__init__("ナイト", 80, 0, 50, 25)


class Wizard(Player):
    def __init__(self):
        super().__init__("魔法使い", 50, 5, 25, 15)

    def magic(self):
        self.magic_name = '氷の魔法'
        self.mp -= 1
        amount_of_damage = 40
        return amount_of_damage


class Slime(Monster):
    def __init__(self):
        super().__init__("スライム", 20, 0, 25, 15)


class Mummy(Monster):
    def __init__(self):
        super().__init__("ミイラ", 60, 0, 30, 20)


class Dragon(Monster):
    def __init__(self):
        super().__init__("ドラゴン", 100, 3, 45, 30)

    def special(self):
        self.special_name = '炎を吐く'
        self.mp -= 1
        amount_of_damage = 25
        return amount_of_damage


class DemonKing(Monster):
    def __init__(self):
        super().__init__("魔王", 200, 5, 50, 30)

    # return True: Used special method
    def special(self, players):
        self.special_name = '雷'
        amount_of_damage = 20
        if self.mp > 0 and random.random() < 1/4:
            self.mp -= 1
            print(f'{self.name} の攻撃！ {self.name} は {self.special_name} を使った.')
            for player in players:
                player.hp -= amount_of_damage
                print(f'{player.name} は {amount_of_damage} のダメージを受けた．')
                if player.hp <= 0:
                    print(f'{player.name} は気絶した．')
            return True
        else:
            return False

    def attack(self, target):
        amount_of_damage = self.power - target.defense // 2
        target.hp -= amount_of_damage
        print(f'{self.name} の攻撃！ {target.name} は {amount_of_damage} のダメージを受けた．')

        if target.hp <= 0:
            print(f'{target.name} は気絶した．')
            return True
        else:
            return False




