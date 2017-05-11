# -*- coding: utf-8 -*-
# Generated by Django 1.9.1 on 2017-05-11 01:50
from __future__ import unicode_literals

from django.db import migrations, models
import webapp.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImageClassification',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('modified_date', models.DateTimeField(auto_now=True)),
                ('status', models.CharField(choices=[('0', 'INACTIVE'), ('1', 'ACTIVE')], default='1', help_text='Category Status', max_length=2)),
                ('image_file', models.FileField(blank=True, null=True, upload_to=webapp.models.generate_filename)),
                ('result', models.TextField(blank=True, help_text='Search Result', null=True)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
