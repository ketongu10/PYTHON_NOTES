PATH = "/home/popovpe/SERVICES/JangaFX.LiquidGen_0.3.0/LiquidGen/bldr/pipes/Flow_from_elbow_side.fbx"
EXPORT_DIR = "/home/popovpe/SERVICES/JangaFX.LiquidGen_0.3.0/LiquidGen/results/fbxs_new"
PATH_W_PIPES = "/home/popovpe/SERVICES/JangaFX.LiquidGen_0.3.0/LiquidGen/bldr/pipes"
PATH2ANIMS = "/home/popovpe/SERVICES/JangaFX.LiquidGen_0.3.0/LiquidGen/experiments/animations_for_pipes_cp"
PATH_CUBE = "/home/popovpe/SERVICES/JangaFX.LiquidGen_0.3.0/LiquidGen/bldr/cube.fbx"
BASE_ANIM_PATH_006 = "/home/popovpe/SERVICES/JangaFX.LiquidGen_0.3.0/LiquidGen/experiments/anim_template_006.liquigen"
BASE_ANIM_PATH_003 = "/home/popovpe/SERVICES/JangaFX.LiquidGen_0.3.0/LiquidGen/experiments/anim_template_003.liquigen"
N_target_min, N_target_max = 3, 7
N_speed_min, N_speed_max = 3, 7
GENERAL_DELAY = 0.3
LARGE_DELAY = 3
ANIMATION_PER_PIPE = 10
pos = {
    "Project.open": (217, 228),
    "Project.anim_template": (314,264),
    "Project.filename": (379, 475),
    "LiquiGen.icon": (1602, 782),
    "StopAnim": (140, 1049),
    "Exit": (1900, 40),
    "Exit.dont_save": (840, 620),
    "Emitter.shapes": (1204, 484),
    "Collider.shapes": (1209, 902),
    "ExpMesh": (1650, 634),
    "ExpMesh.dir": (1345, 284),
    "ExpMesh.filename": (1320, 315),
    "ExpMesh.format_slider": (1485, 315),
    "ExpMesh.fbx_pos": (1485, 363),
    "ExpMesh.export_now": (1285, 343),
    "ExpMesh.cancel": (1124, 605),
    "Import": (1002, 349),
    "Import.pos": (1082, 578),
    "Import.filepath": (1242, 800),
    "Import.reimport": (1250, 832),
    "Import.select_all": (1522, 440),
    "Import.select_none": (1625, 440),
    "Import.elements_x": 1222,
    "Import.transform_after_sh_up": (1351, 233),
    "Import.scale": (1250, 375),
    "Import.create": (1041, 706),
    "Import.cp_pos_1": (1050, 576),
    "Import.cp_pos_2": (1000, 250),
    "Import.geometry": (1108, 290),
    "Shtorka.right": (1827, 260),
    "Shtorka.down": (1378, 1007),
    "Shtorka.up": (1427, 117),
    "Shtorka.up.win": (1427, 125),
    "Shtorka.left.down": (384, 1012),
    "Shtorka.left.mid": (384, 540),
    "Anim.set_value.x": 300, #(300, 666),
    "Anim.x.start": 392,
    "Anim.x.end": 1147,
    "Anim.y": 660,
    "Emitter.TargetPos.Toggle": (996, 662),
    "Emitter.TargetPos.x": (1262, 662),
    "Emitter.TargetPos.y": (1412, 662),
    "Emitter.TargetPos.z": (1568, 662),
    "Anim.TargetPos.x": 294,
    "Anim.TargetPos.y.x": 707,
    "Anim.TargetPos.y.y": 732,
    "Anim.TargetPos.y.z": 753,
    "File": (),
    "File.SaveAs": (),


}
class Info:

    def __init__(self, name, v=(0, 10), target={"x":  (-10, 10), "y": (-10, 10), "z": 100}, sphere_pos_y=466, hidden_pos_y=(None,), resolution=None):
        self.name = name
        self.v = v
        self.target = target
        self.sphere_pos_y = sphere_pos_y
        self.hidden_pos_y = hidden_pos_y
        self.anim = "anim_tmpl"
        self.resolution = resolution

    def get_hidden_in_collision(self):
        return [self.sphere_pos_y]+list(self.hidden_pos_y)

pipes = {
    #"Vjuh_from_pipe": Info("Vjuh_from_pipe", sphere_pos_y=572),
    "Flow_from_pipe": Info("Flow_from_pipe", v=(2, 10)),
    "Flow_from_pipe_crack": Info("Flow_from_pipe_crack", resolution=0.03, v=(1, 10), target={"x":  100, "y": (-20, 20), "z": (0, 50)}),
    "Flow_from_elbow": Info("Flow_from_elbow", v=(4, 10)),
    "Flow_from_elbow_side":Info ("Flow_from_elbow_side", hidden_pos_y=(605,), v=(2, 10),target={"x":  100, "y": (-20, 20), "z": (0, 50)}),
    "Flow_from_spider": Info("Flow_from_spider", v=(1, 10), target={"x":  (-2, 2), "y": (-2, 2), "z": 0}),
    "Flow_from_spider_large": Info("Flow_from_spider_large", v=(0, 10), target={"x":  (-10, 10), "y": (-10, 10), "z": (-10, 10)}),
    "Flow_from_flance": Info("Flow_from_flance", hidden_pos_y=(547,), resolution=0.03, v=(0, 10), target={"x":  (-10, 10), "y": (-10, 10), "z": (-10, 10)}),
    "Flow_from_half_flance": Info("Flow_from_half_flance", v=(5, 15)),
    "Flow_from_kran": Info("Flow_from_kran", v=(5, 10)),
}
