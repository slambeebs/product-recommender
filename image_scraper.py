from icrawler.builtin import GoogleImageCrawler
import os

# Mapping: StockCode → Product Description
products = {
    "22384": "LUNCH BAG PINK POLKADOT",
    "22727": "ALARM CLOCK BAKELIKE RED",
    "22112": "CHOCOLATE HOT WATER BOTTLE",
    "23298": "SPOTTY BUNTING",
    "20726": "LUNCH BAG WOODLAND",
    "21034": "REX CASH+CARRY JUMBO SHOPPER",
    "21931": "JUMBO STORAGE BAG SUKI",
    "22139": "RETROSPOT TEA SET CERAMIC 11 PC",
    "22077": "6 RIBBONS RUSTIC CHARM",
    "22423": "REGENCY CAKESTAND 3 TIER"
}

output_dir = 'cnn_data'

os.makedirs(output_dir, exist_ok=True)

for stock_code, query in products.items():
    folder = os.path.join(output_dir, stock_code)
    os.makedirs(folder, exist_ok=True)

    crawler = GoogleImageCrawler(storage={"root_dir": folder})
    crawler.crawl(keyword=query, max_num=30)  # Increase if needed
