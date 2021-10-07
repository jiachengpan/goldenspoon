import goldenspoon

db = goldenspoon.Database('data')

print('## FUND')
for k, v in db.fund_stats.items():
    print('###', k)
    print(v.columns)

print('## STOCK')
for k, v in db.stock_stats.items():
    print('###', k)
    print(v.columns)
