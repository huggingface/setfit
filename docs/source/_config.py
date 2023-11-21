# docstyle-ignore
INSTALL_CONTENT = """
# SetFit installation
! pip install setfit
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/setfit.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]