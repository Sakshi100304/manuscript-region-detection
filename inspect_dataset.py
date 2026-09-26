import os

SOURCE = "dataset/extracted/Dataset"

for split in ["train", "val", "test"]:
    image_dir = os.path.join(SOURCE, split, "images")
    gt_dir = os.path.join(SOURCE, split, "GT")

    print("\n" + "=" * 50)
    print(split.upper())

    if os.path.exists(image_dir):
        images = os.listdir(image_dir)
        print("Images:", len(images))
        print("First 10 images:")
        for f in images[:10]:
            print("  ", f)
    else:
        print("Images folder not found")

    if os.path.exists(gt_dir):
        gt = os.listdir(gt_dir)
        print("GT:", len(gt))
        print("First 10 GT files:")
        for f in gt[:10]:
            print("  ", f)
    else:
        print("GT folder not found")