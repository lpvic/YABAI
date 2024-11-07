{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: __init__

   {% block methods %}

   {%- if methods %}
   {%- if not ((methods|length == 1) and (methods[0] == '__init__')) %}
   .. rubric:: {{ _('Methods') }}
   {%- endif %}

   .. autosummary::
   {%- for item in methods %}
      {%- if item != '__init__' %}
        ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {%- for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}
