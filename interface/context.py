from typing import Dict, Optional
from machinable.group import Group
from machinable.element import normversion
from machinable import get
from machinable.utils import is_directory_version

class Context(Group):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active = []
    
    def add(self, module, version, **kwargs):
        if module in [
            "machinable.index",
            "machinable.project",
        ] and is_directory_version(version):
            # interpret as shortcut for directory
            version = {"directory": version}
        c = self.load_file('context.json', default=[])
        c.append({
            'module': module, 
            'version': normversion(version), 
            **kwargs
        })
        self.save_file("context.json", c)
        
    def code(self):
        pass
        
    def __enter__(self):
        for ctx in self.load_file('context.json', default=[]):
            element = get(**ctx)
            self._active.append(element)
            element.__enter__()
            
    def __exit__(self, *args, **kwargs):
        for ctx in self._active:
            ctx.__exit__()