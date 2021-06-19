<!-- https://cardiospike.ip3.ru/uploadrg/uploadrg.aspx       multipart/form-data ???-->





<form action="/uploadrg/" method="POST" enctype="multipart/form-data">
<p>id		<input type="text" name="id" value="testuser" /></p>

<p>
<!--<a href="test\test.0.rhythmogram.csv">скачать тестовая маленькая файлик с ритмограммой</a><br/>-->
rgfile	<input type="file" name="rgfile"/></p>

<p>rgtext	<textarea name="rgtext" xcols="40" xrows="3">
time,x,y
600;0;0
800;0;0
</textarea></p>
<p>version	<input type="text" name="version" value="cardiospike01" /></p>
<p><input type="submit"/></p>
</form>

