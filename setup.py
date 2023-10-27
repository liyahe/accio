# -*- coding:utf-8 -*-
from distutils.core import setup
from setuptools import find_packages

setup(name='accio',
        version='0.0.1',
        packages=find_packages(),  # 查找包的路径
        # package_dir={'accio'},  # 包的root路径映射到的实际路径
        include_package_data=True,
        description='A python lib.',
        long_description='long_description',
        author='liyahe',
        author_email='',
        # url='http://www.xxxxx.com/',  # homepage
        license='MIT',
        install_requires=[],
        )
# name : 打包后包的文件名
# version : 版本号
# author : 作者
# author_email : 作者的邮箱
# py_modules : 要打包的.py文件
# packages: 打包的python文件夹
# include_package_data : 项目里会有一些非py文件,比如html和js等,这时候就要靠include_package_data 和 package_data 来指定了。package_data:一般写成{‘your_package_name’: [“files”]}, include_package_data还没完,还需要修改MANIFEST.in文件.MANIFEST.in文件的语法为: include xxx/xxx/xxx/.ini/(所有以.ini结尾的文件,也可以直接指定文件名)
# license : 支持的开源协议
# description : 对项目简短的一个形容
# ext_modules : 是一个包含Extension实例的列表,Extension的定义也有一些参数。
# ext_package : 定义extension的相对路径
# requires : 定义依赖哪些模块
# provides : 定义可以为哪些模块提供依赖
# data_files :指定其他的一些文件(如配置文件),规定了哪些文件被安装到哪些目录中。如果目录名是相对路径,则是相对于sys.prefix或sys.exec_prefix的路径。如果没有提供模板,会被添加到MANIFEST文件中。