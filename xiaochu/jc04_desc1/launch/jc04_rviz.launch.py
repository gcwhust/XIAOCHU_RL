import os
import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包目录
    pkg_path = get_package_share_directory('jc04_desc1')
    file_format = 'xacro'
    if file_format == 'urdf':
        # # urdf文件路径
        default_urdf_path = os.path.join(pkg_path, 'urdf', 'jc04.urdf')###针对urdf
    elif file_format == 'xacro':
        # xacro文件路径
        default_urdf_path = os.path.join(pkg_path, 'urdf', 'jc04.xacro')###针对xacro

    # 声明一个urdf目录的参数，方便修改
    action_declare_arg_model_path = launch.actions.DeclareLaunchArgument(
        name='model',
        default_value=str(default_urdf_path),
        description='加载的模型文件路径'
    )

    #### 通过文件路径，获取内容，并转换成参数值对象，以供传入robot_state_publisher
    if file_format == 'urdf':
        substitutions_command_result = launch.substitutions.Command(['cat ',launch.substitutions.LaunchConfiguration('model')])###针对urdf
    elif file_format == 'xacro':
        substitutions_command_result = launch.substitutions.Command(['xacro ',launch.substitutions.LaunchConfiguration('model')])###针对xacro
    robot_description_value = launch_ros.parameter_descriptions.ParameterValue(substitutions_command_result, value_type=str)

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_value}]
    )

    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
    )

    # 后来保存、增加的rviz默认配置文件路径
    default_rviz_config_path = os.path.join(pkg_path, 'config', 'display_jc04_model.rviz')
    
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', default_rviz_config_path], # 根据保存的默认配置文件加载配置
    )
    
    return launch.LaunchDescription([
        action_declare_arg_model_path,
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node
    ])
