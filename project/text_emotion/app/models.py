from django.db import models

# Create your models here.

class Text(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField('文本', default='', max_length=1000)
    type = models.CharField('类别', default='123', max_length=50)
    create_time = models.DateTimeField('创建时间', auto_now_add=True)
    modify_time = models.DateTimeField('最后修改时间', auto_now=True)
    owner = models.CharField('角色', default='', max_length=50)

    def __str__(self):
        return self.title

    class Meta:
        db_table = 'text'

