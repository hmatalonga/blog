<template>
  <Layout :show-logo="false">
    <!-- Author intro -->
    <Author :show-title="true" />

    <div class="container">
      <h1 class="title">Hi there! I'm Hugo Matalonga.</h1>
      <h2 class="subtitle">I'm a Full-stack developer and a Machine Learning enthusiast. 💻</h2>
      <p class="paragraph">I work as a freelancer and I also do academic research on topic of Green Computing focused on mobile devices.</p>
      <p class="paragraph">
        I am an avid fan of <a href="//github.com/hmatalonga">open-source</a>,
        you can check one of my main projects called <a href="//greenhubproject.org">GreenHub</a>.
      </p>
      <p class="paragraph">Follow me on <a href="//twitter.com/hmatalonga">Twitter</a>.</p>
      <p class="paragraph"><span class="icon">📬</span><a href="mailto:hello@hmatalonga.com">Drop me a line.</a></p>
    </div>

  </Layout>
</template>

<script>
export default {
  metaInfo: {
    title: 'About'
  }
}
</script>

<style lang="scss">
.container {
  display: flex;
  flex-direction: column;
  max-width: var(--content-width);
  margin: 0 auto;

  .title {
    margin-bottom: 30px;
  }

  .subtitle {
    font-weight: 300;
    line-height: 2.5rem;
    margin-top: 0;
    margin-bottom: 30px;
  }

  .paragraph {
    font-size: 1.4rem;
    font-weight: 300;
    margin-bottom: 30px;
    a {
      color: var(--title-color);
      font-weight: 600;
    }
    .icon {
      margin-right: .5rem;
    }
  }
}
</style>
