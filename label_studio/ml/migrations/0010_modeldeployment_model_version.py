"""
新增 ModelDeployment.model_version 欄位。
對應 Triton 模型倉庫版本子目錄（1/, 2/, 3/...），允許部署時指定要載入的版本，預設為 1。
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml', '0009_modeldeployment_instance_group_count'),
    ]

    operations = [
        migrations.AddField(
            model_name='modeldeployment',
            name='model_version',
            field=models.PositiveSmallIntegerField(
                default=1,
                help_text='Triton 模型版本號，對應倉庫目錄下的版本子目錄（例如 1/ 2/ 3/），預設為 1',
            ),
        ),
    ]
