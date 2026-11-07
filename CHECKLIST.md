# What YOU must do yourself (screenshots + commands) - per sub-task
1.1 `python --version; pip list | grep -Ei "tensorflow|keras|mlflow"; python -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))"`
1.2 run src/audit.py -> screenshot of printed tables
1.3 run src/eda.py -> 4 figures + 2-3 sentence captions
1.4 DVC toy (new folder):
    git init && dvc init && echo "a,b\n1,2" > d.csv && dvc add d.csv && git add d.csv.dvc .gitignore && git commit -m v1
    edit d.csv; dvc add d.csv; git commit -am v2; git checkout HEAD~1 -- d.csv.dvc; dvc checkout
    dvc remote add -d origin <url>; dvc push; rm -rf d.csv .dvc/cache; dvc pull
    Then dvc add data/raw for the real dataset. Explain: .dvc file = md5 hash + size + path (a pointer); Git is for text diffs, not GBs.
2.x  run training; screenshots: model.summary(), curves, metric table, error grid; src/tune.py + labs/toy_tuner.py
3.1  python labs/toy_mlflow.py 0.001 / 0.01 ; mlflow ui  (compare 2 runs)   3.2 repeat with .env set; `git check-ignore .env`
3.3-3.6 DagsHub run pages (manual vs --autolog), Artifacts tab, comparison table, registry screenshot, load_champion output
4.x  failure-case screenshots: .txt upload, >5MB file, renamed fake.png
6.1 docker run -d --name n -p 8080:80 -v $PWD:/usr/share/nginx/html nginx; docker logs n; docker exec -it n sh; docker ps -a; docker rm -f n; docker rmi nginx
6.2 `docker compose exec web whoami` (shows appuser)
6.3 `docker compose ps` (healthy) + from another machine `nc -zv <host> 5432` fails
6.4 `docker images; docker history visionguard-web` before/after (e.g. drop --no-cache-dir -> compare)
6.5 break POSTGRES_PASSWORD in .env, `docker compose logs`, fix
6.6 toy repo: labs/test_toy.py + labs/toy_workflow/toy-ci.yml (put in .github/workflows/); secrets live in GitHub Settings > Secrets
7   Postman: POST <API_URL>/predict, header X-API-Key, form-data key `image` (File). Test from phone hotspot.
8   Upload 5+ images via Flask UI -> screenshot results + /history
Report front matter must DISCLOSE AI assistance.
