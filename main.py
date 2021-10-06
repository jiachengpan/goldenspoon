import goldenspoon

db = goldenspoon.Database()

print('## FUND')
for k, v in db.fund_stats.items():
    print('###', k)
    print(v.columns)

print('## STOCK')
for k, v in db.stock_stats.items():
    print('###', k)
    print(v.columns)
