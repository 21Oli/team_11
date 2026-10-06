# Visualization Pack — Figure Captions

## fig01_missingness.png
Shows the share of missing or invalid values, including -999 sentinels, across the three raw tables. The figure highlights which source columns required the most cleaning attention.

## fig02_before_after_cleaning.png
Compares selected variable distributions before and after cleaning. The takeaway is whether the cleaning rules materially changed the observed distributions.

## fig03_yield_distribution.png
Shows the distribution of yield across crop types. Differences in the shape, center, and spread indicate that crop type is an important descriptive dimension of yield.

## fig04_region_crop_heatmap.png
Shows mean yield for every region × crop combination, with plot counts annotated. The strongest and weakest combinations identify important geographic and crop-specific differences.

## fig05_correlation_heatmap.png
Shows correlations between yield, numeric plot-level variables, and weather-derived features. Strong correlations identify candidate drivers for further modeling investigation but do not by themselves establish causality.

## fig06_climate_by_region.png
Shows monthly average temperature across regions and highlights the growing-season window. Regional climate patterns provide context for differences in observed crop performance.

## fig07_yield_vs_season_temp.png
Shows yield against growing-season mean temperature separately for each crop. The fitted trends or point patterns indicate whether each crop has a potential temperature range associated with higher observed yield.

## fig08_price_trends.png
Shows average price per quintal by crop from 2021 to 2024. The figure makes differences in price growth between crops visible.

## fig09_revenue_by_crop_region.png
Shows estimated revenue per hectare by crop and region using yield and market price. This combines production performance with market value to identify economically important crop-region combinations.

## fig10_model_comparison.png
Compares validation RMSE across the models evaluated in the modeling stage, including the mean-predictor baseline. Lower RMSE indicates better predictive accuracy on the held-out validation data.

## fig11_predicted_vs_actual_residuals.png
Shows predicted versus actual yield and the residual pattern for validation observations. The plots help reveal systematic bias, spread, and observations that are difficult for the final model to predict.

## fig12_feature_importance.png
Shows the most important features according to permutation importance. Weather-derived features are visually identified so their contribution to model predictions can be assessed.
