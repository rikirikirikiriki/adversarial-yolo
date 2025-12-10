from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "inria/Train/pos"
        self.lab_dir = "inria/Train/pos/yolo-labels"
        self.cfgfile = "cfg/yolo.cfg"
        self.weightfile = "weights/yolo.weights"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 300
        self.image_size = 640
        self.max_epoch = 1
        self.max_lab = 200

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 2

        self.loss_target = lambda obj, cls: obj * cls


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'

class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"

class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls




class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 8
        self.patch_size = 300

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

# 在patch_config.py末尾新增MyVisDroneConfig配置类，继承自BaseConfig
class MyVisDroneConfig(BaseConfig):
    """ 基于VisDrone数据集的对抗贴纸训练配置 """
    def __init__(self):
        super().__init__()  # 调用基类构造，设置默认参数
        self.img_dir = "data/my-visdrone/images"
        self.lab_dir = "data/my-visdrone/labels"
    
        self.weightfile = "weights/yolo11x-visdrone.pt"
        #self.printfile = "non_printability/30values.txt"
        self.patch_size = 300

        self.start_learning_rate = 0.03

        self.patch_name = 'my-visdrone'
        self.max_tv = 0.165
        self.image_size = 640
        self.img_size=640

        self.batch_size = 2

        self.loss_target = lambda obj, cls: obj

        #子类新增的属性
        self.num_classes = 10                               # VisDrone数据集类别数（10类，与VisDrone官方一致）
        self.class_names = ["pedestrian","people","bicycle","car","van",
                             "truck","tricycle","awning-tricycle","bus","motor"]  # VisDrone类别名称列表
        self.target_class = -1      # 攻击目标类别ID；-1表示不固定特定类别（对所有类别目标通用）
        # VisDrone 场景目标密集，保持较小的 max_lab 可避免在 PatchTransformer 中为过多目标复制 patch 造成显存溢出。
        self.max_lab = 40
        self.max_epoch = 1
        self.print_interval = 10    # 日志打印间隔（每多少个iteration输出一次loss）
        self.save_interval = 50     # Patch保存间隔（每多少epoch保存一次当前贴纸图像）

patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj,
    "my-visdrone": MyVisDroneConfig
}

