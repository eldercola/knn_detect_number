# knn_detect_written_number
**train.csv** contains the label and the data for every training picture(a row is a picture)  
The data in every line is look like this:  
```label``` ```pixel0-pixel1783```, where label means the written number in this picture and pixel is the value of that pixel, ranged from ```0-255```  
**test.csv** contains the data for every training picture(a row is a picture)  
```pixel0-pixel1783```, the pixel is same as that in ```train.csv```  
**truth.csv** is the label set for the data in test.csv.  
```label```, the correct label of each row in ```test.csv```  
code is in ```knn_detect_number.py```  
The method I took in this lab is KNN(K-Nearest_Neighbors).  
