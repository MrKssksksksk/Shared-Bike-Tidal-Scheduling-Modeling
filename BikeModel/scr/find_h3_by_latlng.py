import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from h3_utils import latlng_to_h3

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python scr/find_h3_by_latlng.py <纬度> <经度> [--sys wgs84|gcj02]")
        print("例如: python scr/find_h3_by_latlng.py 22.656 114.035")
        sys.exit(1)
    
    lat = float(sys.argv[1])
    lng = float(sys.argv[2])
    coord_sys = "wgs84"
    
    if len(sys.argv) > 3 and sys.argv[3] == "--sys":
        coord_sys = sys.argv[4] if len(sys.argv) > 4 else "wgs84"
    
    h3_id = latlng_to_h3(lat, lng, coordinate_system=coord_sys)
    print(f"输入坐标: ({lat}, {lng})")
    print(f"坐标系: {coord_sys}")
    print(f"H3 ID (分辨率8): {h3_id}")
    print(f"\n✅ 可用于 prediction_target.json 的 region_h3_list")
