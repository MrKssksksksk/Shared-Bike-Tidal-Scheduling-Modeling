import time
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置区域 
APP_KEY = "XXXXXXXXX" # 替换为你的appKey，通常是一个字符串

START_DATE = "2021-01-01"
END_DATE = "2021-08-31"

BLACKLIST_DATES = [
    # "2021-03-10",  # 格式：YYYY-MM-DD
]

ROWS = 4000  # 每页条数（接口上限）
MAX_PAGE = 5000  # 每天最多抓5000页（防止死循环）
MAX_ROWS_PER_DAY = 10000000  # 每天最多1000万条（异常数据保护）

# 数据异常临界值
MIN_REASONABLE = 500
MAX_REASONABLE = 10000000

SLEEP_TIME = 0.3  # 防封间隔

MAX_WORKERS = 5  # 并发线程数，建议3-5

# 路径配置
BASE_DIR = Path("data/raw")
BASE_DIR.mkdir(parents=True, exist_ok=True)


# 日期生成器（新增黑名单过滤）
def generate_dates(start, end, blacklist):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    blacklist_set = set(blacklist)  # 使用集合提高查找效率

    current = start_dt
    while current <= end_dt:
        readable_date = current.strftime("%Y-%m-%d")
        if readable_date not in blacklist_set:
            yield current.strftime("%Y%m%d"), readable_date
        current += timedelta(days=1)


# 核心抓取函数（单天）
def fetch_one_day(date_str, readable_date):
    """
    date_str: 20210101
    readable_date: 2021-01-01
    """
    output_file = BASE_DIR / f"{readable_date}.csv"

    # 断点续跑
    if output_file.exists():
        print(f"⏭ [Thread-{threading.current_thread().ident}] 已存在，跳过 {readable_date}")
        return True

    print(f"🚀 [Thread-{threading.current_thread().ident}] 开始抓取 {readable_date}")

    url = "https://opendata.sz.gov.cn/api/29200_00403627/1/service.xhtml"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    page = 1
    total_count = 0
    last_page_hash = None
    all_data_rows = []  # 暂存所有数据，最后统一写入

    while True:
        params = {
            "appKey": APP_KEY,
            "page": page,
            "rows": ROWS,
            "startDate": date_str,
            "endDate": date_str,
        }

        if page > MAX_PAGE:
            print(f"⚠️ [Thread-{threading.current_thread().ident}] {readable_date} 达到最大页数，强制停止")
            break

        if total_count > MAX_ROWS_PER_DAY:
            print(f"⚠️ [Thread-{threading.current_thread().ident}] {readable_date} 数据异常过大，停止")
            break

        try:
            res = requests.get(url, headers=headers, params=params, timeout=10)
        except Exception as e:
            print(f"❌ [Thread-{threading.current_thread().ident}] {readable_date} 请求异常，重试中... {e}")
            time.sleep(2)
            continue

        if res.status_code != 200:
            print(f"❌ [Thread-{threading.current_thread().ident}] {readable_date} 状态码错误: {res.status_code}")
            break
        
        response_json = res.json()
        if not response_json or 'data' not in response_json:
            print(f"❌ [Thread-{threading.current_thread().ident}] {readable_date} 响应格式异常，停止")
            break

        data = response_json.get("data", [])

        if not data:
            print(f"✅ [Thread-{threading.current_thread().ident}] {readable_date} 数据结束")
            break
        
        current_hash = hashlib.md5(str(data[:10]).encode()).hexdigest()

        if current_hash == last_page_hash:
            print(f"⚠️ [Thread-{threading.current_thread().ident}] {readable_date} 检测到重复页，停止")
            break

        last_page_hash = current_hash

        df = pd.DataFrame(data)

        # 只保留关键字段
        keep_cols = [
            "START_TIME",
            "END_TIME",
            "START_LAT",
            "START_LNG",
            "END_LAT",
            "END_LNG",
        ]

        df = df[[c for c in keep_cols if c in df.columns]]
        
        # 将当前页数据添加到暂存列表
        all_data_rows.extend(df.to_dict('records'))

        count = len(df)
        total_count += count

        print(f"📄 [Thread-{threading.current_thread().ident}] {readable_date} 第{page}页 | {count}条 | 累计{total_count}")

        if count < ROWS:
            print(f"🎉 [Thread-{threading.current_thread().ident}] {readable_date} 完成，总计 {total_count} 条")
            break

        page += 1
        time.sleep(SLEEP_TIME)

    # 数据校验与写入
    final_df = pd.DataFrame(all_data_rows) if all_data_rows else pd.DataFrame(columns=["START_TIME"])
    
    if final_df.empty:
        print(f"⚠️ [Thread-{threading.current_thread().ident}] {readable_date} 无有效数据，创建空文件")
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        return True
    
    # 写入CSV（一次性写入，避免追加模式可能导致的中断问题）
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 校验文件是否完整
    if final_df.shape[0] < MIN_REASONABLE or final_df.shape[0] > MAX_REASONABLE:
        print(f"⚠️ [Thread-{threading.current_thread().ident}] {readable_date} 数据量异常 ({final_df.shape[0]} 条)，删除不完整文件")
        if output_file.exists():
            output_file.unlink()
        return False
    else:
        print(f"✅ [Thread-{threading.current_thread().ident}] {readable_date} 数据校验通过，共 {final_df.shape[0]} 条")
        return True


# 主程序
if __name__ == "__main__":
    print(f"🚀 开始多线程抓取任务，线程数: {MAX_WORKERS}")
    dates = list(generate_dates(START_DATE, END_DATE, BLACKLIST_DATES))
    print(f"📊 总计需要处理 {len(dates)} 天数据 (已过滤黑名单)")

    success_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_date = {executor.submit(fetch_one_day, date_str, readable_date): readable_date 
                         for date_str, readable_date in dates}
        
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                    print(f"✅ {date} 抓取成功")
                else:
                    failure_count += 1
                    print(f"❌ {date} 抓取失败或数据异常")
            except Exception as exc:
                failure_count += 1
                print(f'❌ {date} 抛出异常: {exc}')
    
    print(f"\n🎯 全部完成！成功: {success_count}, 失败: {failure_count}")
    print(f"📁 数据保存在: {BASE_DIR.absolute()}")