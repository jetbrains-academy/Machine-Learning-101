Теперь нам осталось только реализовать функцию, которая перекрасит изображение с использованием нужного количества цветов.

### Задание

Реализуйте функцию `recolor(image, n_colors)`, принимающую на вход изображение в виде numpy-массива и количество цветов. Функция должна возвращать массив изображения, цвета которого были изменены в результате работы метода `k_means`. Заготовка для функции находится в файле `processing.py`. Функция `process_image` отвечает за открытие изображения, вызов `recolor` и сохранение файла.

Для сохранения изображения мы сначала создадим объект ```Pillow.Image``` (с ним мы уже сталкивались в задании **Чтение изображения**), воспользовавшись методом ```fromarray```, [создающим](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray) изображение из массива. Теперь необходимо сохранить изображение, используя метод ```image.save```.


<div class="column" style="float: left;width: 45%;padding: 5px;">
    <img src="superman-batman.png" alt="Исходное изображение" style="width:100%">
    <p style="text-align:center;">Исходное изображение</p>
</div>
<div class="column" style="float: left;width: 45%;padding: 5px;">
    <img src="superman-batman-after.png" alt="16-цветное изображение" style="width:100%">
    <p style="text-align:center;">8-цветное изображение</p>
</div>
