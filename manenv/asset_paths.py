import os 

class AssetPath:
    """
    Helper class. Contains no logic, only paths to various assets
    
    
    """
    SPAWNER = os.path.join(os.curdir, "assets\\Spawner.png")
    ASSEMBLER = os.path.join(os.curdir, "assets\\Assembler.png")
    PRODUCT_ASSETS = [
        "",
        os.path.join(os.curdir, "assets\\Red_Resource.png"),
        os.path.join(os.curdir, "assets\\Blue_Resource.png")
    ]

    GRABBER = os.path.join(os.curdir, "assets\\Arm.png")
    WELDER = os.path.join(os.curdir, "assets\\Welder.png")
    DISCARD = os.path.join(os.curdir, "assets\\Discard.png")

    def get_product_asset(id : int) -> str:
        if (id < 0 or id >= len(AssetPath.PRODUCT_ASSETS)):
            return ""
        return AssetPath.PRODUCT_ASSETS[int(id)]