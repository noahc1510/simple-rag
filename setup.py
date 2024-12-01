from setuptools import setup, find_packages

setup(
    naem='simplerag',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'langchain-community',
        'langchain-openai',
        'langchain-chroma',
        'opencv-python',

        # For ali_text_splitter
        'modelscope',
        'beautifulsoup4',
        'rapidocr_paddle',
        'paddlepaddle'
        # 'addict',
        # https://github.com/modelscope/modelscope/issues/1050
        # 'datasets<=3.0.1',
        # 'oss2',
        # 'simplejson',
        # 'modelscope[nlp]',
    ],
    # dependencies_links=[
    #     'https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html',
    # ]
)
