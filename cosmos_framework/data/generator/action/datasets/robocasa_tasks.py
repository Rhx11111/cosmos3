# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboCasa task lists, importable without the LeRobot reader."""

# All 18 ``target/atomic`` tasks. Base-motion prevalence measured over 40 episodes each:
# NavigateKitchen is the only genuinely mobile task (100% of episodes, 2.6 m mean net
# displacement); PickPlaceDrawerToCounter moves in 48% (<=0.71 m); five more move in
# 2-10% of episodes (<=0.4 m); the remaining ten never move the base (<6 mm noise).
# Training on this set requires ``use_base_action=True`` so the base channels exist.
DEFAULT_ALL_ATOMIC_TASKS: tuple[str, ...] = (
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
)


# The 65 ``pretrain/atomic`` tasks (every atomic task RoboCasa365 ships a pretrain split for;
# the 18 ``target/atomic`` names above are a subset). Human demos only -- the registry also
# lists machine-generated ``mg`` paths for 60 of these, but they are not part of the local
# download. Same 12D action / 16D state / 3-camera contract as target/atomic, so this loader
# handles them unchanged; only ``root`` and ``task_names`` differ.
DEFAULT_PRETRAIN_ATOMIC_TASKS: tuple[str, ...] = (
    "AdjustToasterOvenTemperature", "AdjustWaterTemperature", "CheesyBread", "CloseBlenderLid",
    "CloseCabinet", "CloseDishwasher", "CloseDrawer", "CloseElectricKettleLid", "CloseFridge",
    "CloseFridgeDrawer", "CloseMicrowave", "CloseOven", "CloseStandMixerHead",
    "CloseToasterOvenDoor", "CoffeeServeMug", "CoffeeSetupMug", "LowerHeat", "MakeIcedCoffee",
    "NavigateKitchen", "OpenBlenderLid", "OpenCabinet", "OpenDishwasher", "OpenDrawer",
    "OpenElectricKettleLid", "OpenFridge", "OpenFridgeDrawer", "OpenMicrowave", "OpenOven",
    "OpenStandMixerHead", "OpenToasterOvenDoor", "PackDessert", "PickPlaceCabinetToCounter",
    "PickPlaceCounterToBlender", "PickPlaceCounterToCabinet", "PickPlaceCounterToDrawer",
    "PickPlaceCounterToMicrowave", "PickPlaceCounterToOven", "PickPlaceCounterToSink",
    "PickPlaceCounterToStandMixer", "PickPlaceCounterToStove", "PickPlaceCounterToToasterOven",
    "PickPlaceDrawerToCounter", "PickPlaceFridgeDrawerToShelf", "PickPlaceFridgeShelfToDrawer",
    "PickPlaceMicrowaveToCounter", "PickPlaceSinkToCounter", "PickPlaceStoveToCounter",
    "PickPlaceToasterOvenToCounter", "PickPlaceToasterToCounter", "PreheatOven",
    "SlideDishwasherRack", "SlideOvenRack", "SlideToasterOvenRack", "StartCoffeeMachine",
    "TurnOffMicrowave", "TurnOffSinkFaucet", "TurnOffStove", "TurnOnBlender",
    "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet", "TurnOnStove",
    "TurnOnToaster", "TurnOnToasterOven", "TurnSinkSpout",
)

# The 235 ``pretrain/composite`` tasks: multi-stage, long-horizon kitchen activities.
# Episodes average ~1565 frames (78 s) against ~142 for atomic, so by frame count composite
# outweighs pretrain/atomic roughly 18:1 (27.6M vs 1.5M). Mixing the two unweighted therefore
# yields a ~95% composite corpus -- intentional here, but worth remembering when reading
# per-task numbers. Composite demos drive the mobile base far more than atomic ones, which is
# why they require ``use_base_action=True``.
DEFAULT_PRETRAIN_COMPOSITE_TASKS: tuple[str, ...] = (
    "AddIceCubes", "AddLemonToFish", "AddMarshmallow", "AddSugarCubes", "AddSweetener",
    "AdjustHeat", "AfterwashSorting", "AirDryFruit", "AlcoholServingPrep", "AlignSilverware",
    "ArrangeBreadBowl", "ArrangeBuffetDessert", "ArrangeDrinkware", "ArrangeTeaAccompaniments",
    "ArrangeUtensilsByType", "ArrangeVegetables", "AssembleCookingArray", "BalancedMealPrep",
    "BeverageOrganization", "BeverageSorting", "BlendIngredients", "BlendMarinade",
    "BowlAndCup", "BreadAndCheese", "BreadSetupSlicing", "BuildAppetizerPlate", "ButterOnPan",
    "CandleCleanup", "CerealAndBowl", "ChooseMeasuringCup", "ChooseRipeFruit", "CleanBoard",
    "CleanMicrowave", "ClearClutter", "ClearCuttingBoard", "ClearFoodWaste", "ClearFreezer",
    "ClearReceptaclesForCleaning", "ClearSink", "ClearSinkArea", "ClearSinkSpace",
    "CollectWashingSupplies", "ColorfulSalsa", "CondimentCollection", "CookieDoughPrep",
    "CoolBakedCake", "CoolKettle", "CreateChildFriendlyFridge", "CupcakeCleanup",
    "CutBuffetPizza", "DateNight", "DefrostByCategory", "DeliverBrewedCoffee", "DeliverStraw",
    "DessertAssembly", "DessertUpgrade", "DisplayMeatVariety", "DistributeChicken",
    "DivideBasins", "DivideBuffetTrays", "DrainVeggies", "DrinkwareConsolidation", "DryDishes",
    "DryDrinkware", "DumpLeftovers", "FillBlenderJug", "FillKettle", "FilterMicrowavableItem",
    "FoodCleanup", "FreezeBottledWaters", "FreezeCookedFood", "FreezeIceTray",
    "FreshProduceOrganization", "FryingPanAdjustment", "GarnishCupcake", "GatherCuttingTools",
    "GatherMarinadeIngredients", "GatherVegetables", "GetToastedBread", "HeatMug",
    "HeatMultipleWater", "HotDogSetup", "JuiceFruitReamer", "KettleBoiling",
    "LemonSeasoningFish", "LineUpCondiments", "LoadCondimentsInFridge", "LoadDishwasher",
    "LoadFridgeByType", "LoadFridgeFifo", "LoadPreparedFood", "MakeFruitBowl",
    "MakeLoadedPotato", "MatchCupAndDrink", "MaximizeFreezerSpace", "MealPrepStaging",
    "MeatSkewerAssembly", "MeatTransfer", "MicrowaveCorrectMeal", "MicrowaveDefrostMeat",
    "MicrowaveThawing", "MicrowaveThawingFridge", "MixCakeFrosting", "MixedFruitPlatter",
    "MoveFreezerToFridge", "MoveFridgeToFreezer", "MoveToCounter", "MoveToFreezerDrawer",
    "MultistepSteaming", "OrganizeBakingIngredients", "OrganizeCleaningSupplies",
    "OrganizeCoffeeCondiments", "OrganizeCondiments", "OrganizeMetallicUtensils",
    "OrganizeMugsByHandle", "OrganizeVegetables", "OvenBroilFish", "PackFoodByTemp",
    "PackFruitContainer", "PackIdenticalLunches", "PastryDisplay", "PlaceBeveragesTogether",
    "PlaceDishesBySink", "PlaceEqualIceCubes", "PlaceFoodInBowls", "PlaceIceInCup",
    "PlaceMeatInMarinade", "PlaceMicrowaveSafeItem", "PlaceStraw", "PlaceVegetablesEvenly",
    "PlaceVeggiesInDrawer", "PlateSteakMeal", "PlateStoreDinner", "PortionFruitBowl",
    "PortionInTupperware", "PortionOnSize", "PortionYogurt", "PreRinseStation", "PreSoakPan",
    "PreheatPot", "PrepForSanitizing", "PrepForTenderizing", "PrepFridgeForCleaning",
    "PrepMarinatingMeat", "PrepSinkForCleaning", "PrepareBroilingStation",
    "PrepareCheeseStation", "PrepareCocktailStation", "PrepareCoffee", "PrepareDishwasher",
    "PrepareDrinkStation", "PrepareSausageCheese", "PrepareSmoothie", "PrepareSoupServing",
    "PrepareStoringLeftovers", "PrepareToast", "PressChicken", "PrewashFoodAssembly",
    "PrewashFoodSorting", "QuickThaw", "RearrangeFridgeItems", "RecycleBottlesBySize",
    "RecycleSodaCans", "RecycleStackedYogurt", "RefillCondimentStation", "ReheatMeal",
    "RemoveBroiledFish", "RemoveCuttingBoardItems", "ReorganizeFrozenVegetables",
    "ResetCabinetDoors", "RestockBowls", "RestockCannedFood", "RestockPantry",
    "RestockSinkSupplies", "RetrieveIceTray", "RetrieveMeat", "ReturnHeatedFood",
    "ReturnWashingSupplies", "RinseBowls", "RinseCuttingBoard", "RinseSinkBasin", "RotatePan",
    "SanitizePrepCuttingBoard", "ScalePortioning", "ScrubBowl", "ScrubCuttingBoard",
    "SearingMeat", "SeasoningSpiceSetup", "SeasoningSteak", "ServeSteak", "ServeTea",
    "ServeWarmCroissant", "SetBowlsForSoup", "SetUpCuttingStation", "SetUpSpiceStation",
    "SetupBowls", "SetupButterPlate", "SetupFruitBowl", "SetupFrying", "SetupSodaBowl",
    "SetupWineGlasses", "ShakePan", "SimmeringSauce", "SizeSorting", "SnackSorting",
    "SoakSponge", "SortingCleanup", "SpicyMarinade", "StackBowlsCabinet", "StackBowlsInSink",
    "StackCans", "StartElectricKettle", "SteamInMicrowave", "StirVegetables",
    "StockingBreakfastFoods", "StoreDumplings", "StoreLeftoversByType", "StoreLeftoversInBowl",
    "StrainerSetup", "SweetSavoryToastSetup", "SweetenCoffee", "SweetenHotChocolate",
    "ThawInSink", "TiltPan", "ToastBagel", "ToastBaguette", "ToastOnCorrectRack",
    "TongBuffetSetup", "TransportCookware", "TurnOffSimmeredSauceHeat", "VeggieDipPrep",
    "WarmCroissant", "WashFish", "WashLettuce", "YogurtDelightPrep",
)

# Convenience union for the 300-task pretrain soup (atomic + composite). Both live under one
# converted root, so the loader still takes a single ``root``.
DEFAULT_PRETRAIN_ALL_TASKS: tuple[str, ...] = (
    DEFAULT_PRETRAIN_ATOMIC_TASKS + DEFAULT_PRETRAIN_COMPOSITE_TASKS
)
