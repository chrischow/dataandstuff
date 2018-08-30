
# Test Notebook
This is a notebook to test the export of Jupyter notebooks into HTML format.


```python
# Import modules
import matplotlib.pyplot as plt
import numpy as np

# Settings
%matplotlib inline
plt.style.use('ggplot')
```


```python
# Create variables
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = list(np.array(x) ** 2)
```


```python
# See variables
print("x: " + str(x))
print("y: " + str(y))
```

    x: [1, 2, 3, 4, 5, 6, 7, 8]
    y: [1, 4, 9, 16, 25, 36, 49, 64]
    


```python
# Plot graph
plt.plot(x, y)
plt.title('Test Plot')
plt.show()
```


![png](output_4_0.png)

