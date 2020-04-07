#!/usr/bin/env python
"""turnips installation script."""

import setuptools

def main():
    """turnips installation wrapper"""
    kwargs = {
        'name': 'turnips',
        'version': '0.5.0',
        'author': 'nago',
        'author_email': 'nago@malie.io',
        'description': 'turnips are vegetables with a creamy white color and a purple top.',
        'url': 'https://gitlab.com/nanoNago/turnips',
        'packages': setuptools.find_packages(),
        'classifiers': [
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
        ],
        'setup_requires': [
            'setuptools_scm',
        ],
        'install_requires': [
        ],
        'python_requires': '>=3.7',
	'entry_points': {
            'console_scripts': [
                'turnips = turnips:main',
            ]
        },
    }

    with open("README.rst", "r") as fh:
        kwargs['long_description'] = fh.read()

    setuptools.setup(**kwargs)

if __name__ == '__main__':
    main()
