const BASE_URL = 'https://hmatalonga.com'

export default {
  /*
   ** Nuxt rendering mode
   ** See https://nuxtjs.org/api/configuration-mode
   */
  mode: 'universal',
  /*
   ** Nuxt target
   ** See https://nuxtjs.org/api/configuration-target
   */
  target: 'static',
  /*
   ** Headers of the page
   ** See https://nuxtjs.org/api/configuration-head
   */
  head: {
    titleTemplate: (chunk) => {
      if (chunk) {
        return `${chunk} | Hugo Matalonga`
      }

      return 'Hugo Matalonga'
    },
    htmlAttrs: {
      lang: 'en',
    },
    meta: [
      { charset: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      {
        hid: 'description',
        name: 'description',
        content:
          'My name is Hugo Matalonga, and I am a data scientist and a web developer based in Portugal.',
      },
      // Open Graph
      {
        hid: 'og:site_name',
        property: 'og:site_name',
        content: 'Hugo Matalonga',
      },
      { hid: 'og:type', property: 'og:type', content: 'website' },
      { hid: 'og:url', property: 'og:url', content: BASE_URL },
      // Twitter Card
      {
        hid: 'twitter:card',
        name: 'twitter:card',
        content: 'summary_large_image',
      },
      { hid: 'twitter:site', name: 'twitter:site', content: '@hmatalonga' },
      {
        hid: 'twitter:title',
        name: 'twitter:title',
        content: 'Hugo Matalonga',
      },
    ],
    link: [
      { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
      { rel: 'canonical', href: BASE_URL },
    ],
  },
  /*
   ** Global CSS
   */
  css: [],
  /*
   ** Plugins to load before mounting the App
   ** https://nuxtjs.org/guide/plugins
   */
  plugins: [],
  /*
   ** Auto import components
   ** See https://nuxtjs.org/api/configuration-components
   */
  components: false,
  /*
   ** Nuxt.js dev-modules
   */
  buildModules: [
    // Doc: https://github.com/nuxt-community/eslint-module
    '@nuxtjs/eslint-module',
    // Doc: https://github.com/nuxt-community/nuxt-tailwindcss
    '@nuxtjs/tailwindcss',
    [
      '@nuxtjs/google-analytics',
      {
        id: 'UA-49971884-1',
      },
    ],
  ],
  /*
   ** Nuxt.js modules
   */
  modules: [
    // Doc: https://github.com/nuxt/content
    '@nuxt/content',
    '@nuxtjs/feed',
  ],
  /*
   ** Content module configuration
   ** See https://content.nuxtjs.org/configuration
   */
  content: {
    liveEdit: false,
    markdown: {
      prism: {
        theme: 'prism-themes/themes/prism-material-oceanic.css',
      },
    },
  },

  feed() {
    const baseUrlArticles = 'https://hmatalonga.com/blog'
    const { $content } = require('@nuxt/content')

    const createFeedArticles = async function (feed) {
      feed.options = {
        title: 'Hugo Matalonga',
        description:
          'My name is Hugo Matalonga, and I am a data scientist and a web developer based in Portugal.',
        link: baseUrlArticles,
      }
      const articles = await $content('blog').fetch()

      articles.forEach((article) => {
        const url = `${baseUrlArticles}/${article.slug}`

        feed.addItem({
          title: article.title,
          id: url,
          link: url,
          description: article.description,
          content: article.description,
          author: 'Hugo Matalonga',
        })
      })

      feed.addContributor({
        name: 'Hugo Matalonga',
        email: 'hello@hmatalonga.com',
        link: 'https://hmatalonga.com',
      })
    }

    return [
      {
        path: '/feed.xml',
        create: createFeedArticles,
        cacheTime: 1000 * 60 * 15,
        type: 'rss2',
      },
      {
        path: '/feed.json',
        create: createFeedArticles,
        cacheTime: 1000 * 60 * 15,
        type: 'json1',
      },
    ]
  },

  /*
   ** Build configuration
   ** See https://nuxtjs.org/api/configuration-build/
   */
  build: {},
}
