from django.db import models
from datetime import datetime
import random
import string

COMMON_STATUS=(
              ('0','INACTIVE'),
              ('1','ACTIVE'),
              )

def generate_filename(self,filename):
    now = datetime.now()
    year = now.year
    month = now.month
    date = now.day
    random_string = ''.join(random.choice(string.digits) for i in range(3))
    ext = filename.split(".")
    filename = "%s%s.%s" %(ext[0],random_string,ext[1])
    url="static/uploads/image/%d/%d/%d/%s" %(year,month,date,filename)
    return url

class BaseInfo(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=2,default='1',choices=COMMON_STATUS,help_text='Category Status')
    

    class Meta:
        abstract = True	


class ImageClassification(BaseInfo):
    image_file = models.FileField(upload_to=generate_filename,null=True,blank=True)
    result = models.TextField(null=True,blank=True,help_text='Search Result')
    
    def __unicode__(self):
        return u'%s' %(self.image_file)

