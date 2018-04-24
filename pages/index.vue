<template>
  <section class="container">
    <h1 class="title">
      <span>Hugo</span> Matalonga
    </h1>
    <h2 class="subtitle">
      I code & make things.
    </h2>
    <ul class="links">
      <li><a href="https://github.com/hmatalonga">GitHub</a></li>
      <li><a href="https://www.linkedin.com/in/hmatalonga">LinkedIn</a></li>
      <li><a href="https://twitter.com/hmatalonga">Twitter</a></li>
      <li><a href="mailto:hello@hmatalonga.com">Email</a></li>
    </ul>
    <div class="copyright">
      <p>SVG background patterns by <a href="http://www.steveschoger.com/">@steveschoger</a></p>
    </div>
  </section>
</template>

<script>
export default {
}
</script>

<style>
.container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}

.title {
  font-size: 9.6vmin;
  font-weight: 500;
  color: white;
  letter-spacing: -.1rem;
  margin-bottom: 2vmin;
}

.title span {
  display: block;
  font-size: 19.2vmin;
  line-height: 0.75;
}

.subtitle {
  font-size: 3.2vmin;
  font-weight: 400;
}

.links {
  position: fixed;
  width: 100%;
  bottom: 8.5vh;
  padding: 0;
  list-style: none;
}

.links li {
  display: inline;
  padding: 0 1vmin;
}

.links li:first-child {
  padding-left: 0;
}

.links li:last-child {
  padding-right: 0;
}

.links li > a {
  color: hsl(0, 0%, 84%);
  text-decoration: none;
  transition: color .15s ease-in-out;
}
.links li > a:hover {
  color: white;
}

.copyright {
  position: fixed;
  width: 100%;
  bottom: 1vh;
  padding: 0;
  font-size: 1.6vmin;
}

.copyright a {
  color: hsl(0, 0%, 84%);
  text-decoration: none;
  transition: color .15s ease-in-out;
}

.copyright a:hover {
  color: white;
}
</style>
