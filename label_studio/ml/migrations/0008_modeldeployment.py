# Generated manually for ModelDeployment (deploy user-trained models with API key)

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ml', '0007_auto_20240314_1957'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelDeployment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('api_key', models.CharField(db_index=True, help_text='API Key，呼叫部署 API 時須在 Header X-API-Key 帶入', max_length=64, unique=True)),
                ('is_enabled', models.BooleanField(default=True, help_text='是否啟用此部署；停用後無法以 API Key 呼叫', verbose_name='is_enabled')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='created at')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='updated at')),
                ('ml_backend', models.OneToOneField(help_text='對應的 ML Backend（使用者訓練的模型）', on_delete=django.db.models.deletion.CASCADE, related_name='model_deployment', to='ml.mlbackend')),
            ],
            options={
                'verbose_name': 'Model Deployment',
                'verbose_name_plural': 'Model Deployments',
            },
        ),
    ]
