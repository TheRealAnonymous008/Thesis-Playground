class AssetPath:
    """
    Helper class. Contains no logic, only paths to various assets
    
    
    """
    SPAWNER = "assets\Spawner.png"
    PRODUCT_ASSETS = [
        "",
        "assets\Red_Resource.png",
        "assets\Blue_Resource.png"
    ]

    def get_product_asset(id : int) -> str:
        if (id < 0 or id >= len(AssetPath.PRODUCT_ASSETS)):
            return ""
        return AssetPath.PRODUCT_ASSETS[int(id)]