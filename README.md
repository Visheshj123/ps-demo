# Parameter Server Demo

## Single Server Scenario All Nodes Runs on Single Machine (development)

### Run in local
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python main.py
```

If you run into pytorch installing issue, try to use python 3.8.

e.g.

```bash
# install 3.8.10 on MacOS
brew install pyenv
pyenv install 3.8.10
pyenv global pyenv
```

### Run in docker

```bash
docker build -t ps-demo:1.0 .
docker run ps-demo:1.0
```

## TODO
- [ ] Multiple Servers Scenario
- [ ] Fault Tolerance
- [ ] Split DB
