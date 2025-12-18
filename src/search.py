import argparse
from service import Searcher


def run_search(image_path, limit, level, threshold, config_path):
    # 1. 初始化服务
    searcher = Searcher(config_path=config_path)

    # 2. 一行代码执行搜索
    print(f"[Search] Processing {image_path} ...")
    results = searcher.search(image_path, limit=limit, level=level, threshold=threshold)

    # 3. 打印结果 (View 层逻辑)
    print(f"\n=== 搜索结果 (Found: {len(results)}) ===")

    for i, item in enumerate(results):
        type_icon = "[VIDEO]" if item['type'] == 'video' else "[IMAGE]"
        level_tag = item['data_level'].upper()

        print(f"[{i + 1}] {type_icon} {level_tag} | 相似度: {item['score']}")
        print(f"    - 文件: {item['file_name']}")

        # 使用标准化后的 time_info
        t = item['time_info']
        if t['mode'] == 'range':
            print(f"    - 时段: {t['display']} (Duration: {t['duration_ms']}ms)")
            print(f"    - 最佳帧: {item['frame_id']}")
        elif t['mode'] == 'point':
            print(f"    - 时间: {t['display']}")
            print(f"    - 帧号: {item['frame_id']}")
        else:
            print(f"    - 类型: 静态图片")

        print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--limit", "-n", type=int, default=5)
    parser.add_argument("--level", "-l", default="auto")
    parser.add_argument("--threshold", "-t", type=float, default=0.6)
    parser.add_argument("--config", "-c", default="config.json")
    args = parser.parse_args()

    run_search(args.input, args.limit, args.level, args.threshold, args.config)