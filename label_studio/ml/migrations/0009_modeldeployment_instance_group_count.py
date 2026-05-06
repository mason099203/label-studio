"""
新增 ModelDeployment.instance_group_count 欄位。
對應 Triton instance_group[].count（每個 GPU 並行模型實例數，預設 1，最大 10）。
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml', '0008_modeldeployment'),
    ]

    operations = [
        migrations.AddField(
            model_name='modeldeployment',
            name='instance_group_count',
            field=models.PositiveSmallIntegerField(
                default=1,
                help_text='Triton instance_group count：每個 GPU 並行模型實例數量（1–10）',
            ),
        ),
    ]
