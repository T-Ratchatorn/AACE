import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from utility.cutout import Cutout

import torchvision.transforms.functional as F

class SquarePad:
    def __call__(self, image):
        # Calculate padding
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)

        # Apply padding
        image = F.pad(image, padding, 0, 'constant')
        return image


class food101:
    def __init__(self, batch_size, threads):

#         mean, std = self._get_statistics() #use for normailize
        mean =  [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.Food101(root='./data_food101', split="train", download=True, transform=train_transform)
        test_set = torchvision.datasets.Food101(root='./data_food101', split="test", download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ("apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
                        "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
                        "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
                        "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
                        "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
                        "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
                        "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters",
                        "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib",
                        "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
                        "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
                        "waffles")

    def _get_statistics(self):
        train_set = torchvision.datasets.Food101(root='./Food101', split="train", download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
