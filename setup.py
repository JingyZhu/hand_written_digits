from setuptools import setup

setup(
    name='recog',
    version='0.1.0',
    packages=['recog'],
    include_package_data=True,
    install_requires=[
        'flask==0.12.2',
        'Flask-Testing==0.6.2',
        'pylint==1.8.1',
    ],
)

setup(
    name='networks',
    version='0.1.0',
    packages=['networks'],
    include_package_data=True,
    install_requires=[
        'numpy==1.14.2',
        'Pillow==5.1.0',
        'pylint==1.8.1',
    ],
)