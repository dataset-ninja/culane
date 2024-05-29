import os
import shutil

import supervisely as sly
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s
from dataset_tools.convert import unpack_if_archive


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    dataset_path = "/home/alex/DATASETS/IMAGES/CULane"
    batch_size = 10

    train_split_path = "/home/alex/DATASETS/IMAGES/CULane/list/train.txt"
    val_split_path = "/home/alex/DATASETS/IMAGES/CULane/list/val.txt"
    test_split_path = "/home/alex/DATASETS/IMAGES/CULane/list/test.txt"
    test_categories_folder = "/home/alex/DATASETS/IMAGES/CULane/list/test_split"

    ds_name_to_split = {"train": train_split_path, "val": val_split_path, "test": test_split_path}

    images_ext = ".jpg"
    anns_ext = ".lines.txt"

    lines_ext = "_gt.txt"


    def create_ann(image_path):
        labels = []
        tags = []

        video_value = image_path.split("/")[-2]
        tag_video = sly.Tag(video, value=video_value)
        tags.append(tag_video)

        if ds_name == "test":
            qwert = image_path.split(dataset_path)[-1][1:]
            category_name = test_pathes_to_category_value.get(qwert)
            category_tagmeta = name_to_tag[category_name]
            tag_category = sly.Tag(category_tagmeta)
            tags.append(tag_category)

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = 590  # image_np.shape[0]
        img_wight = 1640  # image_np.shape[1]

        image_subpath = image_path.split(dataset_path)[1]
        curr_lines_order = image_subpath_to_lines_order[image_subpath]

        ann_path = image_path.replace(images_ext, anns_ext)
        with open(ann_path) as f:
            content = f.read().split("\n")
            if len(content) > 1:
                for idx, coords in enumerate(content):
                    if len(coords) > 0:
                        if curr_lines_order[idx] == "1":
                            coords = coords.strip().split(" ")
                            coords = list(map(float, coords))
                            if len(coords) > 0:
                                obj_class = index_to_class[idx]
                                exterior = []
                                for i in range(0, len(coords), 2):
                                    y = int(coords[i + 1])
                                    x = int(coords[i])
                                    # if y < 0:  # use for polygon, but polyline is better
                                    #     y = 0
                                    # if y >= img_height:
                                    #     y = img_height - 1
                                    # if x < 0:
                                    #     x = 0
                                    # if x >= img_wight:
                                    #     x = img_wight - 1
                                    exterior.append([y, x])
                                polygon = sly.Polyline(exterior)
                                label_poly = sly.Label(polygon, obj_class)
                                labels.append(label_poly)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)


    lane1 = sly.ObjClass("lane 1", sly.Polyline)
    lane2 = sly.ObjClass("lane 2", sly.Polyline)
    lane3 = sly.ObjClass("lane 3", sly.Polyline)
    lane4 = sly.ObjClass("lane 4", sly.Polyline)

    index_to_class = {0: lane1, 1: lane2, 2: lane3, 3: lane4}

    video = sly.TagMeta("sequence", sly.TagValueType.ANY_STRING)

    normal = sly.TagMeta("normal", sly.TagValueType.NONE)
    crowd = sly.TagMeta("crowd", sly.TagValueType.NONE)
    hlight = sly.TagMeta("hlight", sly.TagValueType.NONE)
    shadow = sly.TagMeta("shadow", sly.TagValueType.NONE)
    noline = sly.TagMeta("noline", sly.TagValueType.NONE)
    arrow = sly.TagMeta("arrow", sly.TagValueType.NONE)
    curve = sly.TagMeta("curve", sly.TagValueType.NONE)
    cross = sly.TagMeta("cross", sly.TagValueType.NONE)
    night = sly.TagMeta("night", sly.TagValueType.NONE)

    name_to_tag = {
        "normal": normal,
        "crowd": crowd,
        "hlight": hlight,
        "shadow": shadow,
        "noline": noline,
        "arrow": arrow,
        "curve": curve,
        "cross": cross,
        "night": night,
    }
    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[lane1, lane2, lane3, lane4],
        tag_metas=[video, normal, crowd, hlight, shadow, noline, arrow, curve, cross, night],
    )
    api.project.update_meta(project.id, meta.to_json())

    test_pathes_to_category_value = {}
    for curr_test_file in os.listdir(test_categories_folder):
        category_value = get_file_name(curr_test_file).split("_")[-1]
        curr_path = os.path.join(test_categories_folder, curr_test_file)
        with open(curr_path) as f:
            content = f.read().split("\n")
            for idx, row in enumerate(content):
                if len(row) > 0:
                    test_pathes_to_category_value[row.strip()] = category_value

    for ds_name, split_path in ds_name_to_split.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        with open(split_path) as f:
            content = f.read().split("\n")

        images_names = [im_name for im_name in content if len(im_name) > 0]

        lines_order_path = split_path.replace(".txt", lines_ext)

        image_subpath_to_lines_order = {}

        with open(lines_order_path) as f:
            content = f.read().split("\n")
            for curr_data in content:
                if len(curr_data) > 0:
                    curr_data = curr_data.split(" ")
                    image_subpath_to_lines_order[curr_data[0]] = curr_data[2:]

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [dataset_path + image_name for image_name in img_names_batch]

            img_names_batch = [
                get_file_name(im_name.split("/")[-2]) + "_" + im_name.split("/")[-1]
                for im_name in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))
    return project
