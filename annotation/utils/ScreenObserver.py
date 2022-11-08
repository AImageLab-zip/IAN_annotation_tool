from annotation.utils.metaclasses import SingletonMeta


class ScreenObserver(metaclass=SingletonMeta):
    def __init__(self,):
        self.observed_screens = []

    def add(self, screen):
        print(f'added {screen}')
        self.observed_screens.append(screen)

    def remove(self, screen):
        self.observed_screens.remove(screen)

    def update(self,):
        for observed_screen in self.observed_screens:
            observed_screen.show_() # ??
