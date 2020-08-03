import setuptools
from pip.req import parse_requirements

#https://stackoverflow.com/questions/14399534/reference-requirements-txt-for-the-install-requires-kwarg-in-setuptools-setup-py
install_reqs = parse_requirements('requirements.txt')
reqs = [str(ir.req) for ir in install_reqs]

setuptools.setup(
    name="example-pkg-YOUR-USERNAME-HERE", # Replace with your own username
    version="0.0.1",
    author="Habib Bukhari",
    author_email="habib.s.t.bukhari@gmail.com",
    description="Decode2",
    long_description='cool',
    url="https://github.com/DrHB/test",
    packages=['decode2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
   install_requires= reqs,

    python_requires='>=3.7',
)