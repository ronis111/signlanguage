from django.db import models

# Create your models here.
class SignLanguages(models.Model):
    comment=models.CharField(max_length=100)
    sign=models.FileField(upload_to='static/signgif/',null=True,verbose_name="")