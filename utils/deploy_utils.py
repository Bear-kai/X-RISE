import subprocess
import importlib
from pathlib import Path
import os
import yaml
import logging


def get_logger(filename, level='info', fmt=None):
    level_relations = {
		'debug':logging.DEBUG,
		'info':logging.INFO,
		'warning':logging.WARNING,
		'error':logging.ERROR,
		'crit':logging.CRITICAL
	}
    if fmt is None:
        fmt='[%(asctime)s][%(name)s][%(levelname)s][line:%(lineno)d] - %(message)s'
    # logger = logging.getLogger(os.path.basename(filename))
    logger = logging.getLogger(Path(__file__).stem)
    logger.propagate = False
    format_str = logging.Formatter(fmt)
    logger.setLevel(level_relations.get(level))
    sh = logging.StreamHandler()							
    sh.setFormatter(format_str)
    th = logging.FileHandler(filename=filename,mode='a',encoding='utf-8')	
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)
    return logger


def get_ffmpeg(TASK_ENV, video_size):
    ffmpeg = subprocess.Popen(
                [   "ffmpeg", "-y", 
                    "-loglevel", "error", 
                    "-f", "rawvideo",
                    "-pixel_format", "rgb24", 
                    "-video_size", video_size,
                    "-framerate", "10", 
                    "-i", "-",
                    "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264",
                    "-crf", "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
    return ffmpeg


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_robot_cam_config(args, CONFIGS_PATH):
    # robot
    embodiment_type = args.get("embodiment")  
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def _get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        args["embodiment_name"] = str(embodiment_type[0])
    else:
        args["embodiment_name"] = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # camera
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    return args


def print_config_infos(args):
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + args["embodiment_name"])
    print("\n==================================")