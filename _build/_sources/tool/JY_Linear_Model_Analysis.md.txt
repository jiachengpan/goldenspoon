Name|Training Sample|Training Period|Quarter of raw data in each sample|Training Drop point|Training Label|Test Indicator Sample|Test Indicator Period|Test Label|Test Drop Point|Result Dir|Train Test Shape|Training Sample.1|Month 2 Result
-|-|-|-|-|-|-|-|-|-|-|-|-|-|
4 Sample Drop 0.1|4|'2020-09-30' '2020-12-31' '2021-03-31' '2021-06-30' |4|0.1|2021-07, 2021-08, 2021-09|1.0|2020-12-31' '2021-03-31' '2021-06-30' '2021-09-30' |2021-10, 2021-11|0.15|goldenspoon/linear_regression/regress_result/Joyan_4_Sample_Drop_0.1|x_train.shape:(833, 30)
x_test.shape:(841, 30)
y_train.shape:(833,)
y_test.shape:(841,)|833.0|TP:435, FP:397, TN:0, FN:0
4 Sample Drop 0.0|4|2020-09-30' '2020-12-31' '2021-03-31' '2021-06-30' |4|0|2021-07, 2021-08, 2021-09|1.0|2020-12-31' '2021-03-31' '2021-06-30' '2021-09-30' |2021-10, 2021-11|0.15|goldenspoon/linear_regression/regress_result/Joyan_4_Sample_Drop_0.0|x_train.shape:(1743, 30)
x_test.shape:(841, 30)
y_train.shape:(1743,)
y_test.shape:(841,)|1743.0|TP:352, FP:286, TN:0, FN:0
1 Sample Drop 0.1|1|2021-06-30' |4|0.1|2021-07, 2021-08, 2021-09|1.0|2020-12-31' '2021-03-31' '2021-06-30' '2021-09-30' |2021-10, 2021-11|0.15|goldenspoon/linear_regression/regress_result/Joyan_1_Sample_Drop_0.1|x_train.shape:(356, 30)
x_test.shape:(841, 30)
y_train.shape:(356,)
y_test.shape:(841,)|356.0|TP:16, FP:41, TN:186, FN:320
1 Sample Drop 0.0|1|2021-06-30' |4|0|2021-07, 2021-08, 2021-09|1.0|2020-12-31' '2021-03-31' '2021-06-30' '2021-09-30' |2021-10, 2021-11|0.15|goldenspoon/linear_regression/regress_result/Joyan_1_Sample_Drop_0.0|x_train.shape:(588, 30)
x_test.shape:(841, 30)
y_train.shape:(588,)
y_test.shape:(841,)|588.0|TP:5, FP:27, TN:127, FN:227


Indicator|4 Sample Drop 0.1|4 Sample Drop 0.0|1 Sample Drop 0.1|1 Sample Drop 0.0|nan|nan|nan|nan|nan|nan|nan|nan|nan
是|-2.79828735898948|-2.0220749027096|-0.213851623349892|-0.212779603933551|nan|nan|nan|nan|nan|nan|nan|nan|nan
成长型|0.0718076341572599|0.0397850849032057|-0.306892298046677|-0.165474230133382|nan|nan|nan|nan|nan|nan|nan|nan|nan
混合型|-0.189255303714296|-0.0408477541341792|-0.501212575954301|-0.475687294214171|nan|nan|nan|nan|nan|nan|nan|nan|nan
价值型|0.117447669557083|0.0010626692309811|0.808104874000976|0.641161524347557|nan|nan|nan|nan|nan|nan|nan|nan|nan
小盘股|-0.508627367784811|-0.320520309887235|0.219185042753271|0.119688616947795|nan|nan|nan|nan|nan|nan|nan|nan|nan
中盘股|-1.97598381383474|-1.56566538273583|-0.388046353508441|-0.302325868633108|nan|nan|nan|nan|nan|nan|nan|nan|nan
大盘股|2.48461118161957|1.88618569262314|0.168861310755168|0.182637251685311|nan|nan|nan|nan|nan|nan|nan|nan|nan
market_value_mean|0.2973901990956|0.252945606093706|0.98998694137632|0.411673434763462|nan|nan|nan|nan|nan|nan|nan|nan|nan
market_value_std|3.65422547398602|2.12939133168992|0.575860236335145|0.156950851643065|nan|nan|nan|nan|nan|nan|nan|nan|nan
fund_shareholding_mean|-0.11585026339379|-0.0992174879176917|-0.913215213870594|-0.560533099086354|nan|nan|nan|nan|nan|nan|nan|nan|nan
fund_shareholding_std|0.0738007453946605|0.0596326386809867|0.650411878468783|0.394318456049602|nan|nan|nan|nan|nan|nan|nan|nan|nan
fund_number_mean|-0.915138594149963|-0.537437292135064|1.04758313948845|0.537968755794811|nan|nan|nan|nan|nan|nan|nan|nan|nan
fund_number_std|0.843985177672509|0.49947148035828|0.86651053323838|0.770542625328639|nan|nan|nan|nan|nan|nan|nan|nan|nan
close_price_mean|-0.131672880958113|-0.0516187857251925|-0.307987433749335|-0.0380780319026526|nan|nan|nan|nan|nan|nan|nan|nan|nan
close_price_std|0.342683431870966|0.216209438281513|0.398903854309644|0.187275102664412|nan|nan|nan|nan|nan|nan|nan|nan|nan
avg_price_mean|0.0471804682266103|-0.0553340461897235|0.362648239677965|0.0122435787362939|nan|nan|nan|nan|nan|nan|nan|nan|nan
avg_price_std|-0.395014282338132|-0.219273943968455|-0.566700679334488|-0.285260286780238|nan|nan|nan|nan|nan|nan|nan|nan|nan
turnover_rate_mean|-0.0315609795344241|-0.0233528524703005|-0.0462879287921146|-0.0482714918279657|nan|nan|nan|nan|nan|nan|nan|nan|nan
turnover_rate_std|0.0121302505107155|0.00966488698739267|0.029111384120158|0.0258685566578325|nan|nan|nan|nan|nan|nan|nan|nan|nan
amplitutde_mean|-0.00257103244966816|-0.00145439987280764|-0.0875273388943059|-0.0545015197546773|nan|nan|nan|nan|nan|nan|nan|nan|nan
amplitutde_std|-0.0131994561798665|-0.0130272847998439|0.00530812899029765|-0.000184898888985882|nan|nan|nan|nan|nan|nan|nan|nan|nan
margin_diff_mean|-0.00688763424270738|-0.0039093852876413|-0.00757460757189666|-0.00584684682811022|nan|nan|nan|nan|nan|nan|nan|nan|nan
margin_diff_std|0.00212849239628798|-0.00558112254143972|0.0121298823146524|-0.00341167811369905|nan|nan|nan|nan|nan|nan|nan|nan|nan
share_ratio_of_funds_mean|-0.00563891348224758|0.00247443967716368|0.0370222224518471|0.00582781673355548|nan|nan|nan|nan|nan|nan|nan|nan|nan
share_ratio_of_funds_std|0.0226416202933411|0.0123151324002008|-0.00764498974864496|0.0117943027812302|nan|nan|nan|nan|nan|nan|nan|nan|nan
num_of_funds_mean|-0.0293556941338537|-0.00748816193824965|-0.152190612214423|-0.0317729158903231|nan|nan|nan|nan|nan|nan|nan|nan|nan
num_of_funds_std|0.0348378884614054|0.00779559344514676|0.121986146533258|0.0301647878019265|nan|nan|nan|nan|nan|nan|nan|nan|nan
fund_owner_affinity_mean|-0.0286878117354048|-0.0573986214221135|-0.106437074915203|-0.125954106121925|nan|nan|nan|nan|nan|nan|nan|nan|nan
fund_owner_affinity_std|-0.0669050270093916|-0.056789472801097|-0.212187141833982|-0.177737215231028|nan|nan|nan|nan|nan|nan|nan|nan|nan
cyclical_industry|0.0513585787277709|0.0258172302254879|0.112873877056484|0.0536993478297865|nan|nan|nan|nan|nan|nan|nan|nan|nan