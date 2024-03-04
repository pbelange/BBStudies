
# init file
from pathlib import Path
import BBStudies.Tracking.Jobs as Jobs

# Allows access to this job dictionnary
JOBS = {f.name[:4]:str(f) for f in list(Path(Jobs.__file__).parents[0].glob('J*'))} # dict of job path
