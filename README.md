# knn_detect_written_number
**train.csv** contains the label and the data for every training picture(a row is a picture)  
The data in every line is look like this:  
```label``` ```pixel0-pixel1783```, where label means the written number in this picture and pixel is the value of that pixel, ranged from ```0-255```  
**test.csv** contains the data for every training picture(a row is a picture)  
```pixel0-pixel1783```, the pixel is same as that in ```train.csv```  
**truth.csv** is the label set for the data in test.csv.  
```label```, the correct label of each row in ```test.csv```  
code is in ```knn_detect_number.py```  
The method I took in this lab is [KNN(K-Nearest_Neighbors)](https://baike.baidu.com/item/%E9%82%BB%E8%BF%91%E7%AE%97%E6%B3%95/1151153#:~:text=%E9%82%BB%E8%BF%91%E7%AE%97%E6%B3%95%EF%BC%8C%E6%88%96%E8%80%85%E8%AF%B4K%E6%9C%80%E8%BF%91%E9%82%BB%EF%BC%88KNN%EF%BC%8CK-NearestNeighbor%EF%BC%89%E5%88%86%E7%B1%BB%E7%AE%97%E6%B3%95%E6%98%AF%20%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%20%E5%88%86%E7%B1%BB%20%E6%8A%80%E6%9C%AF%20%E4%B8%AD%E6%9C%80%E7%AE%80%E5%8D%95%E7%9A%84%E6%96%B9%E6%B3%95%E4%B9%8B%E4%B8%80%E3%80%82.%20%E6%89%80%E8%B0%93K%E6%9C%80%E8%BF%91%E9%82%BB%EF%BC%8C%E5%B0%B1%E6%98%AFK%E4%B8%AA%E6%9C%80%E8%BF%91%E7%9A%84%E9%82%BB%E5%B1%85%E7%9A%84%E6%84%8F%E6%80%9D%EF%BC%8C%E8%AF%B4%E7%9A%84%E6%98%AF%E6%AF%8F%E4%B8%AA%E6%A0%B7%E6%9C%AC%E9%83%BD%E5%8F%AF%E4%BB%A5%E7%94%A8%E5%AE%83%E6%9C%80%E6%8E%A5%E8%BF%91%E7%9A%84K%E4%B8%AA%E9%82%BB%E8%BF%91%E5%80%BC%E6%9D%A5%E4%BB%A3%E8%A1%A8%E3%80%82.%20%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95%E5%B0%B1%E6%98%AF%E5%B0%86%E6%95%B0%E6%8D%AE%E9%9B%86%E5%90%88%E4%B8%AD%E6%AF%8F%E4%B8%80%E4%B8%AA%E8%AE%B0%E5%BD%95%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB%E7%9A%84%E6%96%B9%E6%B3%95%20%5B1%5D,%E3%80%82.%20%E4%B8%AD%E6%96%87%E5%90%8D.%20k%E6%9C%80%E9%82%BB%E8%BF%91%E5%88%86%E7%B1%BB%E7%AE%97%E6%B3%95.%20%E5%A4%96%E6%96%87%E5%90%8D.%20k-NearestNeighbor.%20%E7%AE%80%20%E7%A7%B0.%20kNN%E7%AE%97%E6%B3%95.). <!--This is Chinese Version of introduction--> 
In KNN algorithm, it is important to calculate the **distance**, in this lab, I just calculate the total difference of 2 vectors which are N dimention.  
Here is the formula:  
![](http://latex.codecogs.com/svg.latex?\sum_{i=1}^{n}{(X_i-Y_i)})
