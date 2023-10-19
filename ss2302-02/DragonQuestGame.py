from characters import Brave

if __name__ == '__main__':
    brave = Brave()
    # Battle with normal monsters
    brave.start_battle()

    print('#'*30)

    # Battle with Demon King
    brave.start_battle_Demon_King()