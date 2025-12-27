
from django.db import models


class UserTable(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField('姓名', default='', max_length=50)
    password = models.CharField('密码', default='123', max_length=50)
    phone = models.CharField('邮箱', default='', max_length=50)
    create_time = models.DateTimeField('创建时间', auto_now_add=True)
    modify_time = models.DateTimeField('最后修改时间', auto_now=True)
    role = models.IntegerField('角色', default=2)
    description = models.TextField('个人描述')

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'user'





