
# Testing Notebook
<script>
  jQuery(document).ready(function($) {

  $(window).load(function(){
    $('#preloader').fadeOut('slow',function(){$(this).remove();});
  });

  });
</script>

<style type="text/css">
  div#preloader { position: fixed;
      left: 0;
      top: 0;
      z-index: 999;
      width: 100%;
      height: 100%;
      overflow: visible;
      background: #fff url('http://preloaders.net/preloaders/720/Moving%20line.gif') no-repeat center center;
  }

</style>

<div id="preloader"></div><script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<div align ="right"><form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form></div>

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


![png](output_6_0.png)

