import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import numpy as np
import pandas as pd

def dot_product(X, Y):
   if len(X) != len(Y):
      return 0
   return sum(i[0] * i[1] for i in zip(X, Y))


def dot(df):
   X = df["order_price"]
   Y = df["order_amount"]
   if len(X) != len(Y):
      return 0
   return sum(i[0] * i[1] for i in zip(X, Y))


def str_to_list(x):
    x1 = x[1:len(x)-1]
    x_new = [float(i) for i in x1.split(", ")]
    return x_new


#--------------
#
#  Sales
#
#--------------

baby_sales = pd.read_csv('shopee_ID_data_sales_Baby_20170820_20171120.csv')
baby_sales.loc[baby_sales.modelid == 0, 'modelid'] = baby_sales.itemid
baby_sales.order_price = baby_sales.order_price.apply(str_to_list)
baby_sales.order_amount = baby_sales.order_amount.apply(str_to_list)
baby_sales["amount_sold"] = baby_sales["order_amount"].apply(sum)
baby_sales["revenue"] = baby_sales.apply(dot, axis=1)
#baby_sales["average_price"] = baby_sales["order_price"].apply(sum) / baby_sales["order_price"].apply(len)
#baby_sales["frequency"] = baby_sales["order_price"].apply(len)
baby_sales = baby_sales[["itemid", "amount_sold", "revenue"]].groupby('itemid').sum().reset_index()
baby_sales["average_price"] = baby_sales["revenue"] / baby_sales["amount_sold"]
baby_sales[["itemid","average_price","amount_sold","revenue"]].to_csv('agg_Baby_sales.csv', sep=',', index=False)


home_sales = pd.read_csv('shopee_ID_data_sales_Home_20170820_20171120.csv')
home_sales.loc[home_sales.modelid == 0, 'modelid'] = home_sales.itemid
home_sales.order_price = home_sales.order_price.apply(str_to_list)
home_sales.order_amount = home_sales.order_amount.apply(str_to_list)
home_sales["amount_sold"] = home_sales["order_amount"].apply(sum)
home_sales["revenue"] = home_sales.apply(dot, axis=1)
home_sales = home_sales[["itemid", "amount_sold", "revenue"]].groupby('itemid').sum().reset_index()
home_sales["average_price"] = home_sales["revenue"] / home_sales["amount_sold"]
home_sales[["itemid","average_price","amount_sold","revenue"]].to_csv('agg_Home_sales.csv', sep=',', index=False)



#--------------
#
#  Views & Clicks
#
#--------------

baby_views = pd.read_csv('shopee_ID_data_view_Baby_20170820_20171120.csv')
baby_views = baby_views.fillna(0)
agg_baby_views = baby_views[["itemid", "views", "clicks"]].groupby('itemid').sum().reset_index()
agg_baby_views.to_csv('agg_Baby_views.csv', sep=',', index=False)


home_views = pd.read_csv('shopee_ID_data_view_Home_20170820_20171120.csv')
home_views = home_views.fillna(0)
agg_home_views = home_views[["itemid", "views", "clicks"]].groupby('itemid').sum().reset_index()
agg_home_views.to_csv('agg_Home_views.csv', sep=',', index=False)



#--------------
#
#  Data
#
#--------------

baby_data = pd.read_csv('shopee_ID_data_Baby_20170820_20171120.csv')
baby_data.loc[baby_data.price_before_discount == 0, 'price_before_discount'] = baby_data.price
baby_data["price_before_discount"] = baby_data["price_before_discount"].fillna(baby_data["price"])

baby_data["average_discount"] = (baby_data["price_before_discount"] - baby_data["price"]) / baby_data["sold"]

discount_baby_data = baby_data[["itemid", "average_discount"]].groupby('itemid').mean().reset_index()

other_baby_data = baby_data[["itemid", "sold","image_count","liked_count",
                       "cmt_count","rating_good","rating_normal","rating_bad"]].groupby('itemid').sum().reset_index()

baby_data = pd.merge(discount_baby_data, other_baby_data, on='itemid', how='outer')
baby_data.to_csv('agg_Baby_data.csv', sep=',', index=False)



home_data = pd.read_csv('shopee_ID_data_Home_20170820_20171120.csv')
home_data.loc[home_data.price_before_discount == 0, 'price_before_discount'] = home_data.price
home_data["price_before_discount"] = home_data["price_before_discount"].fillna(home_data["price"])

home_data["average_discount"] = (home_data["price_before_discount"] - home_data["price"]) / home_data["sold"]

discount_home_data = home_data[["itemid", "average_discount"]].groupby('itemid').mean().reset_index()

other_home_data = home_data[["itemid", "sold","image_count","liked_count",
                       "cmt_count","rating_good","rating_normal","rating_bad"]].groupby('itemid').sum().reset_index()

home_data = pd.merge(discount_home_data, other_home_data, on='itemid', how='outer')
home_data.to_csv('agg_Home_data.csv', sep=',', index=False)


#--------------
#
#  Join all
#
#--------------

baby_combine = pd.merge(baby_sales, agg_baby_views, on='itemid', how='outer')
baby_combine = pd.merge(baby_combine, baby_data, on='itemid', how='outer')
baby_combine.to_csv('Baby_combined.csv', sep=',', index=False)

home_combine = pd.merge(home_sales, agg_home_views, on='itemid', how='outer')
home_combine = pd.merge(home_combine, home_data, on='itemid', how='outer')
home_combine.to_csv('Home_combined.csv', sep=',', index=False)
