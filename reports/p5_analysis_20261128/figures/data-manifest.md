# Figure data manifest

全部真实数据，不插补、不合成。

|Figure|Data|Aggregation|Script|Outputs|
|---|---|---|---|---|
|fig_test_low/nominal/high|data/test_time_curves.csv|60秒非重叠块；碰撞率与完成数累计；先测试重复后训练种子均值±SD|analyze.py|450dpi PNG + SVG|
|fig_density_response|data/seed_density_means.csv|训练种子均值±SD|analyze.py|450dpi PNG + SVG|
|fig_safety_efficiency|data/seed_density_means.csv|各种子及均值|analyze.py|450dpi PNG + SVG|
|fig_all_seed_results|data/seed_equal_density_means.csv|密度等权的全部三种子|analyze.py|450dpi PNG + SVG|
|fig_training_endpoints|data/online_endpoints.csv; offline_final_loss.csv|实际保存末点，不是训练曲线|analyze.py|450dpi PNG + SVG|
