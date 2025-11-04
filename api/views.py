# api/views.py
import os
from django.http import HttpResponse, Http404, JsonResponse # 导入 HttpResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework import viewsets, filters, generics, status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import action
from django_filters.rest_framework import DjangoFilterBackend
import math
from django.conf import settings
from django.db import transaction
from django.db.models import Q, Exists, OuterRef, Prefetch
from .models import (
    Device, ProbabilityDataSet, Profile, Experiment, Parameter, DataPoint, CsvFile, # 导入 CsvFile
    safe_float
)
from .serializers import (
    UserSerializer, DeviceSerializer, DeviceListSerializer,
    ProbabilityDataSetSerializer, ProfileSerializer,
    ExperimentSerializer, GridDataUpdateSerializer, CsvMetadataUpdateSerializer,
    CsvFileSerializer # 导入新的 CsvFileSerializer (稍后创建)
)
from django.contrib.auth.models import User
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.postgres.aggregates import StringAgg

# --- 认证视图 (不变) ---
class UserCreateView(generics.CreateAPIView):
    # ... (代码不变) ...
    permission_classes = [permissions.AllowAny]
    queryset = User.objects.all()
    serializer_class = UserSerializer

# --- 评估与计算视图 (不变) ---
class DamageAssessmentView(APIView):
    # ... (代码不变) ...
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request, *args, **kwargs):
        # ... (代码不变) ...
        try:
            pt_gw = float(request.data.get('pt_gw', 0))
            gt_db = float(request.data.get('gt_db', 0))
            gr_db = float(request.data.get('gr_db', 0))
            f_ghz = float(request.data.get('f_ghz', 0))
            d_km = float(request.data.get('d_km', 0))
            lna_gain_db = float(request.data.get('lna_gain_db', 0))

            if pt_gw <= 0 or f_ghz <= 0 or d_km <= 0:
                return Response({'error': '功率、频率和距离必须为正数'}, status=status.HTTP_400_BAD_REQUEST)

            pt_dbm = 10 * math.log10(pt_gw * 1e9 * 1000)
            f_mhz = f_ghz * 1000
            ls_db = 20 * math.log10(d_km) + 20 * math.log10(f_mhz) + 32.45
            pr_dbm = pt_dbm + gt_db + gr_db - ls_db + lna_gain_db
            risk_level = "低危"
            if pr_dbm > -20:
                risk_level = "高危"
            elif pr_dbm > -40:
                risk_level = "中危"
            limiter_loss_db = 1.5
            response_data = {
                'ls_db': round(ls_db, 3), 'pr_dbm': round(pr_dbm, 3), 'lna_gain_db': lna_gain_db,
                'limiter_loss_db': limiter_loss_db, 'risk_level': risk_level
            }
            return Response(response_data)
        except (ValueError, TypeError) as e:
            return Response({'error': f'输入数据格式错误: {e}'}, status=status.HTTP_400_BAD_REQUEST)


class LinkAssessmentView(APIView):
    # ... (代码不变) ...
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request, *args, **kwargs):
        # ... (代码不变) ...
        try:
            pt2_kw = float(request.data.get('pt2_kw', 0))
            gt2_db = float(request.data.get('gt2_db', 0))
            gr2_db = float(request.data.get('gr2_db', 0))
            f2_ghz = float(request.data.get('f2_ghz', 0))
            d2_km = float(request.data.get('d2_km', 0))
            receiver_sensitivity_dbm = float(request.data.get('receiver_sensitivity_dbm', 0))

            if pt2_kw <= 0 or f2_ghz <= 0 or d2_km <= 0:
                return Response({'error': '功率、频率和距离必须为正数'}, status=status.HTTP_400_BAD_REQUEST)
            pt2_dbm = 10 * math.log10(pt2_kw * 1000 * 1000)
            f2_mhz = f2_ghz * 1000
            lp_db = 20 * math.log10(d2_km) + 20 * math.log10(f2_mhz) + 32.45
            pr2_dbm = pt2_dbm + gt2_db + gr2_db - lp_db
            link_margin_db = pr2_dbm - receiver_sensitivity_dbm
            link_status = "正常" if link_margin_db > 0 else "中断"
            response_data = {
                'lp_db': round(lp_db, 3), 'link_margin_db': round(link_margin_db, 3), 'link_status': link_status
            }
            return Response(response_data)
        except (ValueError, TypeError) as e:
            return Response({'error': f'输入数据格式错误: {e}'}, status=status.HTTP_400_BAD_REQUEST)

class SystemFailureProbabilityView(APIView):
    # ... (代码不变) ...
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request, *args, **kwargs):
        # ... (代码不变) ...
        components_data = request.data.get('components', [])
        pr_dbm_input = safe_float(request.data.get('pr_dbm'))

        if not components_data:
            return Response({'error': '缺少组件数据'}, status=status.HTTP_400_BAD_REQUEST)
        if pr_dbm_input is None:
             return Response({'error': '无效的到靶功率输入'}, status=status.HTTP_400_BAD_REQUEST)

        system_failure_probability = 0
        total_weight = sum(c.get('weight', 0) for c in components_data if c.get('data') and c.get('weight', 0) > 0)

        if total_weight == 0:
            return Response({'system_failure_probability': 0})

        for component in components_data:
            weight = component.get('weight', 0)
            data = component.get('data', [])

            if not data or weight <= 0:
                continue

            points = []
            for row in data:
                power_str = row.get('功率')
                prob_str = row.get('概率')
                power = safe_float(power_str)
                prob = safe_float(prob_str)
                if power is not None and prob is not None:
                    points.append({'power': power, 'probability': prob})

            if not points:
                continue

            points.sort(key=lambda p: p['power'])

            component_probability = 0
            if not points: # 处理空 points 的情况
                 pass
            elif pr_dbm_input <= points[0]['power']:
                component_probability = points[0]['probability']
            elif pr_dbm_input >= points[-1]['power']:
                component_probability = points[-1]['probability']
            else:
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i+1]
                    if p1['power'] <= pr_dbm_input <= p2['power']:
                        if p2['power'] == p1['power']:
                            component_probability = p1['probability']
                        else:
                            component_probability = p1['probability'] + \
                                (pr_dbm_input - p1['power']) * \
                                (p2['probability'] - p1['probability']) / \
                                (p2['power'] - p1['power'])
                        break

            system_failure_probability += (weight / total_weight) * component_probability

        return Response({'system_failure_probability': round(system_failure_probability, 6)})


# --- DeviceViewSet (保持之前的修改) ---
class DeviceViewSet(viewsets.ModelViewSet):
    # ... (代码不变) ...
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = Device.objects.all().order_by('-created_at')
    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    filterset_fields = ['device_type']
    search_fields = ['name', 'device_number', 'tech_description']

    def get_queryset(self):
        queryset = super().get_queryset()
        experiment_type = self.request.query_params.get('experiment_type')
        if experiment_type:
            queryset = queryset.filter(experiments__experiment_type=experiment_type).distinct()

        if self.action == 'list':
            queryset = queryset.annotate(
                _test_types=StringAgg(
                    'experiments__experiment_type',
                    delimiter='/',
                    distinct=True,
                    ordering='experiments__experiment_type'
                )
            )
        elif self.action == 'retrieve':
             # 优化详情视图，预加载完整数据结构
             queryset = queryset.prefetch_related(
                Prefetch('experiments', queryset=Experiment.objects.order_by('created_at').prefetch_related(
                    Prefetch('parameters', queryset=Parameter.objects.order_by('column_index').prefetch_related(
                        Prefetch('datapoints', queryset=DataPoint.objects.order_by('row_index'))
                    ))
                ))
            )
        return queryset

    def get_serializer_class(self):
        if self.action == 'list':
            return DeviceListSerializer
        return DeviceSerializer

    # --- 新增：用于在特定 Device 下创建 Experiment 的 action ---
    # 如果不使用嵌套路由，可以通过这个 action 创建
    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def add_experiment(self, request, pk=None):
        device = self.get_object()
        serializer = ExperimentSerializer(data=request.data)
        if serializer.is_valid():
            # 自动关联 device
            serializer.save(device=device)
            # 处理 grid_data (如果创建时提供了)
            grid_data = request.data.get('grid_data_for_editing') # 假设前端用这个 key
            if grid_data:
                try:
                    self._update_grid_data(serializer.instance, grid_data)
                except Exception as e:
                    # 如果 grid_data 更新失败，可以选择删除刚创建的 experiment 或返回错误
                    serializer.instance.delete()
                    return Response({'error': f'创建实验成功，但更新表格数据失败: {e}'}, status=status.HTTP_400_BAD_REQUEST)

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # 辅助方法：更新 Grid Data (内部调用)
    def _update_grid_data(self, experiment, grid_data):
        """
        根据传入的 grid_data (二维数组) 更新 experiment 的 Parameters 和 DataPoints。
        这是一个覆盖式更新。
        """
        grid_serializer = GridDataUpdateSerializer(data={
            'parameters': [{'name': h, 'column_index': i} for i, h in enumerate(grid_data[0])],
            'datapoints': [
                {'parameter_col_index': c_idx, 'row_index': r_idx, 'value_text': cell}
                for r_idx, row in enumerate(grid_data[1:])
                for c_idx, cell in enumerate(row)
            ]
        })
        grid_serializer.is_valid(raise_exception=True)
        validated_params = grid_serializer.validated_data['parameters']
        validated_dps = grid_serializer.validated_data['datapoints']

        with transaction.atomic():
            # 1. 删除旧的 Parameters 和 DataPoints
            experiment.parameters.all().delete() # DataPoints 会级联删除

            # 2. 创建新的 Parameters
            param_objects = []
            param_map = {} # {column_index: Parameter instance}
            for param_data in validated_params:
                # 注意：这里可能需要从前端获取 unit 信息，或从旧数据恢复
                param = Parameter(experiment=experiment, **param_data)
                param_objects.append(param)
            Parameter.objects.bulk_create(param_objects)

            # 获取新创建的 Parameter 的 ID 并建立映射
            newly_created_params = Parameter.objects.filter(experiment=experiment)
            for param in newly_created_params:
                 param_map[param.column_index] = param

            # 3. 创建新的 DataPoints
            datapoint_objects = []
            for dp_data in validated_dps:
                col_idx = dp_data['parameter_col_index']
                if col_idx in param_map:
                    param_instance = param_map[col_idx]
                    numeric_value = safe_float(dp_data.get('value_text'))
                    datapoint_objects.append(DataPoint(
                        parameter=param_instance,
                        row_index=dp_data['row_index'],
                        value_text=dp_data.get('value_text'),
                        value_numeric=numeric_value
                    ))

            if datapoint_objects:
                DataPoint.objects.bulk_create(datapoint_objects)


# --- 新增：Experiment ViewSet ---
class ExperimentViewSet(viewsets.ModelViewSet):
    queryset = Experiment.objects.all().order_by('-created_at')
    serializer_class = ExperimentSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['device', 'experiment_type']

    def get_queryset(self):
        queryset = super().get_queryset()
        if self.action == 'retrieve':
             queryset = queryset.prefetch_related(
                Prefetch('parameters', queryset=Parameter.objects.order_by('column_index').prefetch_related(
                    Prefetch('datapoints', queryset=DataPoint.objects.order_by('row_index'))
                )),
                Prefetch('csv_files', queryset=CsvFile.objects.order_by('uploaded_at')) # 预加载 CsvFile
            )
        return queryset

    # --- 自定义 Action: 更新 Grid Data (保持不变) ---
    @action(detail=True, methods=['patch'], serializer_class=GridDataUpdateSerializer)
    def grid_data(self, request, pk=None):
        # ... (代码不变) ...
        experiment = self.get_object()
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated_params = serializer.validated_data['parameters']
        validated_dps = serializer.validated_data['datapoints']
        try:
            with transaction.atomic():
                experiment.parameters.all().delete()
                param_objects = []; param_map = {}
                for param_data in validated_params:
                    unit = param_data.get('unit')
                    param = Parameter(experiment=experiment, name=param_data['name'], column_index=param_data['column_index'], unit=unit)
                    param_objects.append(param)
                created_param_instances = Parameter.objects.bulk_create(param_objects)
                for param in created_param_instances: param_map[param.column_index] = param

                datapoint_objects = []
                for dp_data in validated_dps:
                    col_idx = dp_data['parameter_col_index']
                    if col_idx in param_map:
                        param_instance = param_map[col_idx]
                        numeric_value = safe_float(dp_data.get('value_text'))
                        datapoint_objects.append(DataPoint(parameter=param_instance, row_index=dp_data['row_index'], value_text=dp_data.get('value_text'), value_numeric=numeric_value))
                if datapoint_objects: DataPoint.objects.bulk_create(datapoint_objects)
            updated_experiment = self.get_queryset().get(pk=experiment.pk)
            return Response(ExperimentSerializer(updated_experiment).data)
        except Exception as e:
            import traceback; traceback.print_exc()
            return Response({'error': f'更新表格数据失败: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


    # --- 自定义 Action: 更新 CSV Metadata (保持不变) ---
    @action(detail=True, methods=['patch'], serializer_class=CsvMetadataUpdateSerializer)
    def csv_metadata(self, request, pk=None):
        # ... (代码不变) ...
        experiment = self.get_object()
        serializer = self.get_serializer(experiment, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    # --- 新增 Action: 上传 CSV 文件 ---
    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated], serializer_class=CsvFileSerializer)
    def upload_csv(self, request, pk=None):
        print(f"--- upload_csv action received request for experiment pk={pk} ---")  # <--- 添加
        print("Request FILES:", request.FILES)  # <--- 添加
        print("Request POST data:", request.data)  # <--- 添加
        experiment = self.get_object()
        # 'metadata_id' 应该由前端在添加元数据时生成并一起上传
        metadata_id = request.data.get('metadata_id')
        uploaded_file = request.FILES.get('file')

        if not uploaded_file:
            return Response({'error': '未找到上传的文件'}, status=status.HTTP_400_BAD_REQUEST)
        if not metadata_id:
            return Response({'error': '缺少 metadata_id'}, status=status.HTTP_400_BAD_REQUEST)

        # 尝试将 metadata_id 转换为整数
        try:
            metadata_id_int = int(metadata_id)
        except (ValueError, TypeError):
             return Response({'error': '无效的 metadata_id'}, status=status.HTTP_400_BAD_REQUEST)

        # 检查是否已存在具有相同 metadata_id 的文件（避免重复上传）
        if CsvFile.objects.filter(metadata_id=metadata_id_int).exists():
            # 可以选择更新现有文件或返回错误
             return Response({'error': f'元数据ID {metadata_id_int} 已存在关联的文件'}, status=status.HTTP_400_BAD_REQUEST)
             # 或者：
             # existing_csv = CsvFile.objects.get(metadata_id=metadata_id_int)
             # existing_csv.file = uploaded_file
             # existing_csv.save()
             # serializer = self.get_serializer(existing_csv)
             # return Response(serializer.data, status=status.HTTP_200_OK)


        csv_file_instance = CsvFile(
            experiment=experiment,
            file=uploaded_file,
            metadata_id=metadata_id_int
        )
        csv_file_instance.save()

        # 更新 Experiment 的 metadata，将 CsvFile 的 ID 关联进去
        # 这部分逻辑比较复杂，取决于前端如何管理 metadata_id
        # 一个简单的做法是假设前端在调用 csv_metadata 更新时会包含这个 ID
        # 或者在这里查找对应的 metadata 项并更新
        metadata_updated = False
        try:
            current_metadata = list(experiment.csv_files_metadata) # 确保是列表副本
            for item in current_metadata:
                if isinstance(item, dict) and item.get('metadata_id') == metadata_id_int:
                    item['csv_file_id'] = csv_file_instance.id # 关联 CsvFile ID
                    metadata_updated = True
                    break
            if metadata_updated:
                experiment.csv_files_metadata = current_metadata
                experiment.save(update_fields=['csv_files_metadata'])
            else:
                # 如果元数据中还没这个 id，可能需要前端先更新元数据
                print(f"警告: 在 Exp {pk} 的 metadata 中未找到 metadata_id {metadata_id_int}")

        except Exception as meta_e:
            # 如果元数据更新失败，可以选择删除刚上传的文件或记录错误
            print(f"警告: 更新 Experiment {pk} 的 metadata 失败: {meta_e}")
            # csv_file_instance.delete() # 考虑回滚

        serializer = self.get_serializer(csv_file_instance)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    # --- 新增 Action: 获取 CSV 数据 ---
    @action(detail=True, methods=['get'], url_path='csv_data/<int:metadata_id>')
    def get_csv_data(self, request, pk=None, metadata_id=None):
        """
        根据 Experiment ID (pk) 和 CSV 的 metadata_id 获取 CSV 文件内容。
        """
        print(f"--- get_csv_data action called for experiment pk={pk}, metadata_id={metadata_id} ---")  # 添加调试打印
        experiment = self.get_object()  # 获取 Experiment 对象
        try:
            # metadata_id 已经是整数了，无需转换
            # metadata_id_int = int(metadata_id)
            csv_file = get_object_or_404(CsvFile, experiment=experiment, metadata_id=metadata_id)  # 直接使用 metadata_id

            if not csv_file.file or not csv_file.file.storage.exists(csv_file.file.name):
                print(
                    f"--- CsvFile found (id={csv_file.id}), but file does not exist at {csv_file.file.name} ---")  # 添加调试打印
                raise Http404("CSV 文件物理文件未找到")

            print(f"--- Found CsvFile (id={csv_file.id}), attempting to read file: {csv_file.file.name} ---")  # 添加调试打印

            # 读取文件内容... (保持不变)
            try:
                content = csv_file.file.read().decode('utf-8')
            except UnicodeDecodeError:
                csv_file.file.seek(0)
                try:
                    content = csv_file.file.read().decode('gbk')
                except Exception as e:
                    print(f"--- Error decoding file content: {e} ---")  # 添加调试打印
                    return Response({'error': f'无法解码文件内容: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                csv_file.file.close()

            response = HttpResponse(content, content_type='text/csv; charset=utf-8')
            return response

        except CsvFile.DoesNotExist:
            print(f"--- CsvFile.DoesNotExist for experiment={pk}, metadata_id={metadata_id} ---")  # 添加调试打印
            raise Http404(f"找不到 metadata_id 为 {metadata_id} 的 CSV 文件记录")


# --- DeviceComparisonView (保持之前的修改) ---
class DeviceComparisonView(APIView):
    # ... (代码不变) ...
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    def post(self, request, *args, **kwargs):
        # ... (代码不变) ...
        data = request.data
        device_type = data.get('device_type')
        experiment_type = data.get('experiment_type')
        fixed_params = [p for p in data.get('fixed_params', []) if p.get('name')]

        if not device_type or not experiment_type:
            return Response({'error': '必须提供器件类型和实验类型'}, status=status.HTTP_400_BAD_REQUEST)

        base_queryset = Device.objects.filter(
            device_type=device_type,
            experiments__experiment_type=experiment_type
        ).distinct()

        filtered_queryset = base_queryset
        param_validation_errors = {}
        for param_filter in fixed_params:
            param_name = param_filter.get('name')
            min_val_str = param_filter.get('min')
            max_val_str = param_filter.get('max')

            min_val = safe_float(min_val_str) if min_val_str else -float('inf')
            max_val = safe_float(max_val_str) if max_val_str else float('inf')

            if min_val is None and min_val_str:
                param_validation_errors[param_name] = f'最小值 "{min_val_str}" 无效'
            if max_val is None and max_val_str:
                param_validation_errors[param_name] = f'最大值 "{max_val_str}" 无效'
            if min_val is not None and max_val is not None and min_val > max_val:
                 param_validation_errors[param_name] = f'最小值 "{min_val}" 不能大于最大值 "{max_val}"'

            if param_name in param_validation_errors:
                continue

            datapoint_subquery = DataPoint.objects.filter(
                parameter__experiment__device_id=OuterRef('pk'),
                parameter__experiment__experiment_type=experiment_type,
                parameter__name=param_name,
                value_numeric__gte=min_val,
                value_numeric__lte=max_val
            )
            filtered_queryset = filtered_queryset.filter(Exists(datapoint_subquery))

        if param_validation_errors:
            error_message = "; ".join([f'参数 "{k}": {v}' for k, v in param_validation_errors.items()])
            return Response({'error': f'输入参数范围错误: {error_message}'}, status=status.HTTP_400_BAD_REQUEST)

        all_available_params = Parameter.objects.filter(
            experiment__device__in=base_queryset,
            experiment__experiment_type=experiment_type
        ).values_list('name', flat=True).distinct().order_by('name')

        final_devices_qs = filtered_queryset.prefetch_related(
            Prefetch('experiments', queryset=Experiment.objects.filter(experiment_type=experiment_type).prefetch_related(
                Prefetch('parameters', queryset=Parameter.objects.order_by('column_index').prefetch_related(
                    Prefetch('datapoints', queryset=DataPoint.objects.order_by('row_index'))
                ))
            ))
        )

        filtered_devices_data = []
        for device in final_devices_qs:
            experiment = next((exp for exp in device.experiments.all() if exp.experiment_type == experiment_type), None)
            if not experiment: continue

            params_ordered = list(experiment.parameters.all())
            headers = [p.name for p in params_ordered]

            datapoints_by_row_col = {}
            max_row_index = -1
            param_index_map = {p.id: p.column_index for p in params_ordered}

            for param in params_ordered:
                 col_idx = param.column_index
                 for dp in param.datapoints.all():
                    datapoints_by_row_col[(dp.row_index, col_idx)] = dp.value_text
                    if dp.row_index > max_row_index:
                        max_row_index = dp.row_index

            rows_data = []
            for row_idx in range(max_row_index + 1):
                row_list = [''] * len(headers)
                for col_idx in range(len(headers)):
                    value = datapoints_by_row_col.get((row_idx, col_idx))
                    if value is not None:
                        row_list[col_idx] = value
                rows_data.append(row_list)

            reconstructed_grid_data = [headers] + rows_data

            filtered_devices_data.append({
                'id': device.id,
                'name': device.name,
                'device_number': device.device_number,
                'device_specific_data': [{
                    'id': experiment.id,
                    'name': experiment.name,
                    'experiment_type': experiment.experiment_type,
                    'grid_data': reconstructed_grid_data,
                    'csv_files': experiment.csv_files_metadata
                }]
            })

        response_payload = {
            'filtered_devices': filtered_devices_data,
            'available_params': list(all_available_params)
        }
        return Response(response_payload)


# --- ProbabilityDataSetViewSet (不变) ---
class ProbabilityDataSetViewSet(viewsets.ModelViewSet):
    # ... (代码不变) ...
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = ProbabilityDataSet.objects.all().order_by('-created_at')
    serializer_class = ProbabilityDataSetSerializer


# --- ProfileView (不变) ---
class ProfileView(APIView):
    # ... (代码不变) ...
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request, *args, **kwargs):
        # ... (代码不变) ...
        profile, created = Profile.objects.get_or_create(user=request.user)
        default_config = {
            'llm_api_url': getattr(settings, 'DEFAULT_LLM_API_URL', ''),
            'llm_api_key': getattr(settings, 'DEFAULT_LLM_API_KEY', ''),
            'llm_model_name': getattr(settings, 'DEFAULT_LLM_MODEL_NAME', ''),
        }
        serializer = ProfileSerializer(profile)
        response_data = serializer.data
        response_data['default_config'] = default_config
        return Response(response_data)

    def patch(self, request, *args, **kwargs):
        # ... (代码不变) ...
        profile = get_object_or_404(Profile, user=request.user) # 使用 get_object_or_404
        serializer = ProfileSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            default_config = {
                'llm_api_url': getattr(settings, 'DEFAULT_LLM_API_URL', ''),
                'llm_api_key': getattr(settings, 'DEFAULT_LLM_API_KEY', ''),
                'llm_model_name': getattr(settings, 'DEFAULT_LLM_MODEL_NAME', ''),
            }
            response_data = serializer.data
            response_data['default_config'] = default_config
            return Response(response_data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)